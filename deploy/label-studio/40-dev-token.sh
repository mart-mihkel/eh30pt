#!/usr/bin/env bash

set -e ${DEBUG:+-x}

if [ -z "${LABEL_STUDIO_USERNAME:-}" ]; then
    echo "=> Skipping the development API token, LABEL_STUDIO_USERNAME is not set."
    exit 0
fi

echo "=> Provisioning the development Label Studio user..."
label-studio init -q >&2

echo "=> Pinning the development API token..."
python3 /label-studio/label_studio/manage.py shell -c '
import os

from django.contrib.auth import get_user_model
from rest_framework.authtoken.models import Token

username = os.environ["LABEL_STUDIO_USERNAME"]
token = os.environ.get("LABEL_STUDIO_USER_TOKEN", "")

user = get_user_model().objects.get(email=username)
organization = user.active_organization
if organization is None:
    raise RuntimeError(f"{username} has no organization to enable legacy API tokens for")

if len(token) > 5:
    if not Token.objects.filter(user=user).update(key=token):
        Token.objects.create(user=user, key=token)

    print("pinned the development API token on", username)
else:
    print("LABEL_STUDIO_USER_TOKEN is too short, keeping the user token as it is")

organization.jwt.legacy_api_tokens_enabled = True
organization.jwt.save()

print("legacy API tokens enabled for", organization)
'

echo "=> Development API token ready."
