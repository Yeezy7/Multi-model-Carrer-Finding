# 导入 SentencePiece 库：用于无监督训练子词（BPE/Unigram）模型以及后续编码/解码
import sentencepiece as spm

def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    重要说明（官方参数文档可查）：
    https://github.com/google/sentencepiece/blob/master/doc/options.md

    Args:
        input_file：原始语料文件路径
        vocab_size：词汇表大小
        model_name：模型名称
        model_type：模型类型
        character_coverage：字符覆盖率
    """
    input_argument = (
        f'--input={input_file} '
        f'--model_prefix={model_name} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type} '
        f'--character_coverage={character_coverage} '
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    )
    
    # 开始训练，会在当前目录下生成 model_name.model 和 model_name.vocab 两个文件
    spm.SentencePieceTrainer.Train(input_argument)
    
def run():
    # ========== 英文分词器配置 ==========
    en_input = "../data/training-parallel-nc-v13/news-commentary-v13.zh-en.en" # 英文语料：一行一句
    en_vocab_size = 32000           # 英文词汇表大小
    en_model_name = "eng"           # 输出前缀：会生成 eng.model 和 eng.vocab 两个文件
    en_model_type = "bpe"           # 使用 BPE 模型
    en_character_coverage = 1.0  # 英文字符集小 -> 用 1.0

    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)
    
    # ========== 中文分词器配置 ==========
    ch_input = "../data/training-parallel-nc-v13/news-commentary-v13.zh-en.zh" # 中文语料：一行一句
    ch_vocab_size = 32000           # 中文词汇表大小
    ch_model_name = "chn"           # 输出前缀：会生成 chn.model 和 chn.vocab 两个文件
    ch_model_type = "bpe"           # 使用 BPE 模型
    ch_character_coverage = 0.9995  # 中文推荐 0.9995，覆盖率越高，OOV 越少，但模型越大

    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)

def test():
    # 加载并调用已训练好的模型进行编码/解码的示例
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今日抵达夏威夷"

    # 加载中文模型
    sp.Load("chn.model")
    
    # 编码为子词片段（字符串），如 ['_美国', '总统', ....]
    print(sp.EncodeAsPieces(text))
    
    # 编码为 id (整数序列)
    print(sp.EncodeAsIds(text))
    
    # 示例：给定一串 id，解码为原始文本
    a = [12907, 277, 7419, 7318, 18384, 28724]
    
    # 注意：python API 的方法名是 CamelCase: DecodeIds / DecodePieces
    print(sp.DecodeIds(a))

if __name__ == "__main__":
    # run()
    test() # 训练完成后，取消注释以做一次快速功能验证