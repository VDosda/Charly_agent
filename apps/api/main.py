from apps.api.app import create_api_app


def create_app():
    return create_api_app()


app = create_app()
