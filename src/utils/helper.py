def parse_command(user_msg_list: str):
    user_msg_list = user_msg_list.split(" ")
    user_command_key = ["stock_code", "command", "start_date", "end_date"]
    # list comprehension
    user_command = {user_command_key[i]: user_msg_list[i]
                    for i in range(len(user_msg_list))}

    return user_command
