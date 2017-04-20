import os


class MenuItem:
    def __init__(self, menu_text, function):
        self.menu_text = menu_text
        self.function = function


def print_menu(menu_item_list):
    for i in range(len(menu_item_list)):
        print(str(i) + ")", menu_item_list[i].menu_text)
    print()

    action = is_valid_menu_item(min=0, max=len(menu_item_list))
    os.system('cls')
    return action


def is_valid_menu_item(min, max, empty_input_allowed=False):
    action = min - 1
    while action < min:
        action = input("Choose an action: ")
        if empty_input_allowed and action == "":
            return min - 1
        try:
            action = int(action)
        except ValueError:
            print("That's not an int!")
            action = -1
            continue

        if action >= max:
            print("That's a high int!")
            action = min - 1
        elif action < 0:
            print("That's a low int!")
            action = min - 1

        return action
