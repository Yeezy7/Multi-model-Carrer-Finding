import json

d = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}


s = json.dumps(d) # 将字典转换为 JSON 字符串
print(s)

l = [
    {
        "name": "Alice",
        "age": 30,
        "city": "New York"
    },
    {
        "name": "Bob",
        "age": 25,
        "city": "Los Angeles"
    },
    {
        "name": "Charlie",
        "age": 35,
        "city": "Chicago"
    }
]

s = json.dumps(l) # 将列表转换为 JSON 字符串
print(s)

json_str = '{"name": "Alice", "age": 30, "city": "New York"}'
json_arrary_str = '[{"name": "Alice", "age": 30, "city": "New York"}, {"name": "Bob", "age": 25, "city": "Los Angeles"}, {"name": "Charlie", "age": 35, "city": "Chicago"}]'

res_dict = json.loads(json_str) # 将 JSON 字符串转换为字典
print(res_dict)
print(type(res_dict))

res_list = json.loads(json_arrary_str) # 将 JSON 字符串转换为列表
print(res_list)
print(type(res_list))