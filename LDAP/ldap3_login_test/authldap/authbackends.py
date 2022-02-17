# auth_demo/authldap/authbackends.py
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.models import User
import ldap3

# 替换为实际的域控 IP
LDAP_HOST = 'xx.xx.xx.xx'


class LdapBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None):
        if ldap_auth(username, password):
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                user = User(username=username)
                if username.endswith('admin'):
                    user.is_staff = True
                    user.is_superuser = True
                user.save()
            return user
        return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None


# @example.com 改为自己域环境的域名
def ldap_auth(username, password):
    username = username + '@example.com'\
        if '@' not in username else username
    server = ldap3.Server(LDAP_HOST, port=636, use_ssl=True)
    conn = ldap3.Connection(server, username, password)
    return conn.bind()