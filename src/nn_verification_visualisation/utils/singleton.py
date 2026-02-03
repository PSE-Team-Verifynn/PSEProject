class SingletonMeta(type):
    '''
    MetaClass that converts its derived classes to singletons.
    Works by interfering the constructor call of the child classes and returning the previously created instance if available.
    :param _instances: dict that stores the instance of every singleton. Maps the type of the subclass to its instance.
    '''
    _instances = {}

    # catches every constructor call of subclasses and prevents object creation if an instance already exists.
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]