
MYCONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'myformatter': {
            "format":
                '\033[0;33m'
                '%(asctime)s'
                '\033[0m'
                ' - ['
                '\033[0;31m'
                "%(levelname).4s"
                '\033[0m'
                '] - '
                '\033[0;32m'
                '%(message)s'
                '\033[0m',
            # "datefmt": '%Y-%m-%d %H:%M:%S',
        },
        'colorized': dict(
            format=''
                '\033[0;34m'
                '%(message)s'
                '\033[0m',
        ),
    },
    'handlers': {
        'myhandler': {  # the name of handler
            'class': 'logging.StreamHandler', 
            'formatter': 'myformatter', 
        },
        'colorized': {
            "class": 'logging.StreamHandler', 
            "formatter": "colorized",
        },
    },
    "loggers": {
        # __name__: {},
        # '': {},
        # 'pytorch_lightning': {
        #     # 'handlers': [],
        #     'handlers': ['colorized'],
        # },
        'main': {  # the name of logger
            'handlers': ['myhandler'],
            'level': 'DEBUG',  # logging level
            'propagate': False,
        },
    },
}