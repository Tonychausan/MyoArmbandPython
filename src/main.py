import MenuActions


def menu():
    menu_item_list = MenuActions.create_main_menu()
    action = MenuActions.print_menu(menu_item_list)
    menu_item_list[action].function()


def main():
    menu()


main()
