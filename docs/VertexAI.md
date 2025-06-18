### Using VertexAI

⚠️ _**Note:** The VertexAI LLM was configured to use the environment variable GOOGLE\_APPLICATION\_CREDENTIALS which contains the file name holding the credentials for your application._

Vertex AI includes both **stable** and **experimental/preview** models. Experimental and preview models may change or be discontinued without notice, so for **production applications**, it’s strongly recommended to use stable models. Check the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs) for the latest information on model status.

#### Prerequisites
- Ensure you have a [Google Cloud project](https://console.cloud.google.com/projectcreate) with [billing enabled](https://console.cloud.google.com/billing).

#### About Authentication
To use Vertex AI programmatically, you’ll create a **service account** and use its credentials to authenticate your application. These credentials, distinct from your personal Google Account credentials, determine your application’s access to Google Cloud services and APIs.

#### Steps

#### 1. Enable the Vertex AI API
- Go to the [Google Cloud API Library](https://console.cloud.google.com/apis/library).
- Select your project from the dropdown at the top.
- Search for "Vertex AI API" and click **Enable**.

#### 2. Create and Configure a Service Account
- Navigate to the [Credentials page](https://console.cloud.google.com/apis/credentials).
- Click **Create Credentials** > **Service Account**.
- Fill in the details (e.g., name and description).
- **Assign roles**: For Vertex AI, grant at least the "Vertex AI User" role.
- Click **Done**, then find your service account, click the three dots (⋮), and select **Manage Keys**.
- Click **Add Key** > **Create New Key**, select **JSON**, and click **Create**.
- The JSON key file will download. **Store it securely**—you won’t be able to download it again.

#### 3. Add the JSON Key File to Your Project
- Place the downloaded JSON file in the **root directory** of your project (e.g., `/my-project/service-account-key.json`).

** If using Docker, also update `compose.yaml` app-service section to include:**
```
volumes:
    - ./service-account-key.json:/app/service-account-key.json
```
This ensures the credentials are available inside the container at runtime.

#### 4. Set the `GOOGLE_APPLICATION_CREDENTIALS` Environment Variable
- Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the **full path** of your JSON file:
  - **.env**:
    ```
    GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
    ```

  - **Unix-like systems**:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
    ```
  - **Windows**:
    ```cmd
    set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\service-account-key.json
    ```

#### 5. Protect Your Credentials
- Add the JSON file to your `.gitignore` file:
  ```
  service-account-key.json
  ```
- **Keep this file private**, as it grants access to your Google Cloud resources and could lead to **unauthorized usage** or billing.

#### Verify Your Setup
- Test your credentials with:
  ```bash
  gcloud auth activate-service-account --key-file=/path/to/your/service-account-key.json
  gcloud auth list
  ```
  Your service account should appear as active.

#### Production Note
This setup is ideal for development. In production, consider secure alternatives like Google Cloud Secret Manager.
