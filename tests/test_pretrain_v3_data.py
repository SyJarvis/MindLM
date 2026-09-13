import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import prepare_pretrain_v3 as prep


class ByteTokenizer:
    eos_token_id = 256

    def __len__(self):
        return 257

    def __call__(self, texts, add_special_tokens=False):
        assert add_special_tokens is False
        encoded = []
        for text in texts:
            terminated = text.endswith('<eos>')
            encoded.append(list((text[:-5] if terminated else text).encode('utf-8')) + ([256] if terminated else []))
        return {'input_ids': encoded}


class PretrainV3DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mindlm-v3-test-')
        self.root = Path(self.temporary.name)
        self.tokenizer_dir = self.root / 'tokenizer'
        self.tokenizer_dir.mkdir()
        (self.tokenizer_dir / 'config.json').write_text('{}', encoding='utf-8')
        self.input = self.root / 'source.csv'
        self.rows = [[f'独立文档 {i} 中文内容 abcdefghijklmnopqrstuvwxyz', 'minimind' if i % 2 else 'open-perfectblend'] for i in range(40)]

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows=None, header=('text', 'source')):
        with self.input.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.rows if rows is None else rows)

    def args(self, output='result', **overrides):
        args = prep.parse_args([
            '--input-csv', str(self.input), '--tokenizer-path', str(self.tokenizer_dir),
            '--output-dir', str(self.root / output), '--workers', '0', '--max-seq-len', '7',
            '--heldout-ppm', '500000', '--batch-rows', '3', '--batch-chars', '100',
        ])
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def build(self, output='result', **overrides):
        with patch.object(prep, 'load_tokenizer', return_value=ByteTokenizer()):
            return prep.prepare(self.args(output, **overrides))

    def test_exact_dedup_prefix_groups_cross_sources_and_chinese(self):
        prefix = '相同文档前缀用于分组而不删除不同正文。' * 12
        self.rows += [[prefix + '甲结尾', 'minimind'], [prefix + '乙结尾', 'open-perfectblend']]
        self.rows += [['cafe\u0301\r\n带缩进\n  x', 'minimind'], ['café\n带缩进\n  x', 'open-perfectblend']]
        self.rows += [['', 'minimind']]
        self.write_rows()
        with self.input.open('a', encoding='utf-8') as stream:
            stream.write('\n')
        manifest = self.build()
        audit = manifest['audit']
        self.assertEqual(audit['logical_rows_read'], len(self.rows) + 1)
        self.assertEqual(audit['blank_csv_rows'], 1)
        self.assertEqual(audit['exact_duplicates_dropped'], 1)
        self.assertEqual(audit['sources']['minimind']['empty_rows'], 1)
        self.assertEqual(audit['retained_documents'], len(self.rows) - 2)
        index = [json.loads(line) for line in (self.root / 'result/documents.jsonl').read_text().splitlines()]
        near = [record for record in index if record['input_row'] in (41, 42)]
        self.assertEqual(len(near), 2)
        self.assertEqual(near[0]['group_sha256'], near[1]['group_sha256'])
        self.assertEqual(near[0]['split'], near[1]['split'])
        self.assertNotEqual(near[0]['text_sha256'], near[1]['text_sha256'])
        self.assertTrue(any(record['input_row'] == 43 for record in index))
        self.assertFalse(any(record['input_row'] == 44 for record in index))
        for key in ('text_sha256', 'group_sha256'):
            train = {record[key] for record in index if record['split'] == 'train'}
            heldout = {record[key] for record in index if record['split'] == 'heldout'}
            self.assertFalse(train & heldout)
        self.assertEqual(audit['train_heldout_exact_intersection'], 0)
        self.assertEqual(audit['train_heldout_group_intersection'], 0)

    def test_boundary_eos_tail_source_counts_and_dataset_compatibility(self):
        self.rows[0][0] += '<eos>'
        self.write_rows()
        manifest = self.build()
        index = [json.loads(line) for line in (self.root / 'result/documents.jsonl').read_text().splitlines()]
        tokenizer = ByteTokenizer()
        from dataset import PackedPretrainDataset
        for split in ('train', 'heldout'):
            expected, source_stream = [], {}
            for record in index:
                if record['split'] != split:
                    continue
                text, source = self.rows[record['input_row'] - 1]
                ids = tokenizer([text.strip()])['input_ids'][0]
                added = ids[-1] != tokenizer.eos_token_id
                self.assertEqual(record['stream_offset'], len(expected))
                self.assertEqual(record['source_tokens'], len(ids))
                self.assertEqual(record['eos_appended'], added)
                expected += ids + ([tokenizer.eos_token_id] if added else [])
                source_stream[source] = source_stream.get(source, 0) + len(ids) + int(added)
            info = manifest['splits'][split]
            full = len(expected) // 8 * 8
            tokens = np.fromfile(self.root / f'result/{split}.bin', dtype='<u4')
            self.assertEqual(tokens.tolist(), expected[:full])
            self.assertEqual(info['discarded_tail_tokens'], len(expected) - full)
            self.assertEqual(info['packed_tokens'], full)
            self.assertEqual(info['stream_tokens'], len(expected))
            self.assertEqual(info['sha256'], hashlib.sha256(tokens.tobytes()).hexdigest())
            self.assertEqual(sum(value['packed_tokens'] for value in info['source_statistics'].values()), full)
            self.assertEqual({source: value['stream_tokens'] for source, value in info['source_statistics'].items()}, source_stream)
            dataset = PackedPretrainDataset(self.root / f'result/{split}', 7, 257)
            x, y, mask = dataset[0]
            self.assertEqual(x.tolist(), expected[:7])
            self.assertEqual(y.tolist(), expected[1:8])
            self.assertEqual(mask.tolist(), [1] * 7)

    def test_split_and_outputs_do_not_depend_on_batch_size(self):
        self.write_rows()
        first = self.build('first', batch_rows=1)
        second = self.build('second', batch_rows=13, batch_chars=5000)
        self.assertEqual(first['tokenizer']['sha256'], second['tokenizer']['sha256'])
        for name in ('train.bin', 'heldout.bin', 'documents.jsonl'):
            self.assertEqual((self.root / 'first' / name).read_bytes(), (self.root / 'second' / name).read_bytes())

    def test_strict_invalid_input_never_publishes_complete_manifest(self):
        cases = {
            'utf8': b'text,source\n\xff,minimind\n',
            'quote': b'text,source\n"unclosed,minimind\n',
            'columns': b'text,source\ncontent,minimind,extra\n',
            'source': b'text,source\ncontent,unknown-source\n',
            'header': b'text\ncontent\n',
            'empty': b'text,source\n,minimind\n',
        }
        for name, data in cases.items():
            with self.subTest(name=name):
                self.input.write_bytes(data)
                with self.assertRaises((UnicodeDecodeError, ValueError, csv.Error)):
                    self.build(name)
                self.assertFalse((self.root / name / 'manifest.json').exists())
                self.assertEqual(json.loads((self.root / name / 'failed.json').read_text())['status'], 'failed')

    def test_unexpected_row_count_and_invalid_token_are_errors(self):
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Expected'):
            self.build('wrong-count', expected_rows=999)
        bad = ByteTokenizer()
        with patch.object(prep, 'load_tokenizer', return_value=bad), patch.object(ByteTokenizer, '__call__', return_value={'input_ids': [[-1]] * 2}):
            with self.assertRaisesRegex(ValueError, 'Token id outside vocabulary'):
                prep.prepare(self.args('bad-token', batch_rows=2, batch_chars=5000))
        self.assertFalse((self.root / 'bad-token/manifest.json').exists())

    def test_existing_output_is_not_overwritten(self):
        self.write_rows()
        directory = self.root / 'existing'
        directory.mkdir()
        marker = directory / 'marker'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.build('existing')
        self.assertEqual(marker.read_text(), 'keep')

    def test_real_tokenizer_spawn_workers_match_serial(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'hello': 2, 'world': 3}, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='[EOS]', unk_token='[UNK]')
        tokenizer.save_pretrained(self.tokenizer_dir)
        self.rows = [[f'hello world item{i} ' + 'hello world ' * 10, 'minimind' if i % 2 else 'open-perfectblend'] for i in range(40)]
        self.write_rows()
        prep.prepare(self.args('serial', workers=0))
        prep.prepare(self.args('parallel', workers=2))
        for name in ('train.bin', 'heldout.bin', 'documents.jsonl'):
            self.assertEqual((self.root / 'serial' / name).read_bytes(), (self.root / 'parallel' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
