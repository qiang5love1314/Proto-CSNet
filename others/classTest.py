from collections import deque

def isValidEncoding(s):
    mapping = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    queue = deque()

    for char in s:
        if char in mapping.keys():
            queue.append(char)
        elif char in mapping.values():
            if not queue or mapping[queue.popleft()] != char:
                return False
        else:
            return False

    return not queue

# 示例用法
input_string = input("请输入一个字符串：")
if isValidEncoding(input_string):
    print("输入的字符串是有效的编码。")
else:
    print("输入的字符串不是有效的编码。")
