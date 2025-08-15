from transformers import BartForConditionalGeneration


def get_model(device='cpu'):
    """
    加载BART模型用于拼写纠错
    自动将模型移动到指定设备（GPU或CPU）
    """
    # 加载中文BART预训练模型
    model = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')

    # 将模型移动到指定设备
    model = model.to(device)

    return model
