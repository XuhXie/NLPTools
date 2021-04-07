import copy

def split_char(str_, tag='[unk]', relapce=True):
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    output = []
    buffer = ''
    replace_list = []
    for s in str_:
        if s in english or s in english.upper():  # 英文或数字
            buffer += s
        else:  # 中文
            if buffer:
                if len(buffer) > 1 and relapce:
                    output.append(tag)
                    replace_list.append(buffer)
                else:
                    output.append(buffer)
            buffer = ''
            output.append(s)

    if buffer:
        if len(buffer) > 1 and relapce:
            output.append(tag)
            replace_list.append(buffer)
        else:
            output.append(buffer)
    #     if (len(replace_list) < 1):
    #         replace_list = ['[NONE]']
    if relapce:
        return output, replace_list
    return output


def recover(lists_, replaces_, tag='[UNK]'):
    result = []
    for senten, reps in zip(lists_, replaces_):
        temp = copy.deepcopy(senten)
        if len(reps) < 1:
            result.append(temp)
            continue
        count = 0
        for index, item in enumerate(senten):
            if count >= len(reps):
                break
            if item == tag:
                temp[index] = reps[count]
                count += 1
        result.append(temp)
    return result