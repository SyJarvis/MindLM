from transformers import AutoTokenizer

# 加载分词器                                                                                                                                  
tokenizer = AutoTokenizer.from_pretrained('mindlm_tokenizer', trust_remote_code=True)                                                  
                                                                                                                                            
# 基本文本编码                                                                                                                                
text = "你好世界"                                                                                                                             
ids = tokenizer(text).data['input_ids']                                                                                                       
print(ids)  # [151644, 104261, 5511, ...]                                                                                                     
                                                                                                                                            
# 解码                                                                                                                                        
print(tokenizer.decode(ids))                                                                                                                  
                                                                                                                                            
# 聊天模板（SFT用）                                                                                                                           
messages = [                                                                                                                                  
    {"role": "user", "content": "你好"},                                                                                                      
    {"role": "assistant", "content": "你好！"},                                                                                               
]                                                                                                                                             
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)                    