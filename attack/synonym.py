SYNONYM_DICT = {
    "转账": ["汇款", "划账"],
    "核实": ["确认", "核对"],
    "客服": ["工作人员", "专员"],
    "账户": ["账号"],
    "立即": ["马上", "尽快"],
    "验证": ["确认", "校验"]
}


def get_synonyms(word):
    return SYNONYM_DICT.get(word, [])
