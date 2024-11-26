from pydantic import SecretStr


def test_no_auth_secret(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is not set, all requests are allowed"""
    mock_settings.AUTH_SECRET = None
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer any-token"},
    )
    assert response.status_code == 200

    # Should also work without any auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 200


def test_auth_secret_correct(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is set, requests with correct token are allowed"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200


def test_auth_secret_incorrect(mock_settings, mock_agent, test_client):
    """Test that when AUTH_SECRET is set, requests with wrong token are rejected"""
    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post(
        "/invoke",
        json={"message": "test"},
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    # Should also reject requests with no auth header
    response = test_client.post("/invoke", json={"message": "test"})
    assert response.status_code == 401
