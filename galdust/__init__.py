from .dl07spec import DL07specContainer

dl07 = DL07specContainer()
# otherspec = OtherspecContainer()


# ................. This part bellow is for example purpose .................
def main():
    print('This is a library package, however, this script is given for descriptive purposes')
    from os import path
    exec(open(path.join(path.dirname(__file__), 'example-usage.py')).read(), globals())



if __name__ == '__main__':
    main()
