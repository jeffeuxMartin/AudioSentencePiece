import os

def load_cached(PRINTDEBUG):
    def load_cached(cls, obj_name, saved_path, msg="Loading ..."):
        PRINTDEBUG(msg)
        if os.path.isdir(saved_path):
            PRINTDEBUG('    (Using local cache...)')
            obj = cls.from_pretrained(saved_path)
        else:
            PRINTDEBUG('    (Loading pretrained...)')
            obj = cls.from_pretrained(obj_name)
            obj.save_pretrained(saved_path)
        return obj
    return load_cached