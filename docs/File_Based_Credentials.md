# File Based Crendentials

As you develop your agents, you might discover that you have credentials that you need to store on disk that you don't want stored in your Git Repo or baked into your container image.

Examples:
- File based LLM Credentials Files (e.g. Google Vertex)
- Certificates or private keys needed for communication with external APIs


The `privatecredentials/` folder give you a quick place to put these files in development.


## How it works

*Protection*
- The .dockerignore file excludes the entire folder to keep it out of the build process.
- The .gitignore files only allows the `.gitkeep` file -- since git doesn't track empty folders.


*Mounted Volume*

The docker compose file mounts the `privatecredentials/` into the container as `/privatecredentials/`. The running container will have access to the untracked files that you have in your development environment.


*Why Not Use Docker Watch*

The syncing feature of Docker Watch isn't used for these reasons:
- docker watch adheres to the rules in `.dockerignore` and therefore won't see the credentials
- even if it did, docker watch doesn't do an initial sync when the container start and will only sync changes that occur while the service is running


## Suggested Use


For each file based credential, do the following:
1. Put the file (e.g. `example-creds.txt`) into the `privatecredentials/` folder
2. In your `.env` file, create an environment variable for the credential (e.g `EXAMPLE_CREDENTIAL=/privatecredentials/example-creds.txt`) that your agent will use to reference the location at runtime
3. In your agent, use the environment variable wherever you need the path to the credential


### Examples

#### Google Vertex
Google Vertex SDK uses the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to locate your credentials file.

Do the following:
1. Put `service-account-key.json` (or `google-credentials.json`)  into the `privatecredentials/` folder
2. In your `.env` file, define `GOOGLE_APPLICATION_CREDENTIALS=/privatecredentials/service-account-key.json`
3. Vertex SDK automatically references the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.



#### Certificate For Signed Communication with remote API
If your agent calls a remote API that requires a client certificate, your agent will need the public certificate to be available.

For example, let's assume you have a cert named `my_remote_api_certificate.cer`

Do the following:
1. Put `my_remote_api_certificate.cer`  into the `privatecredentials/` folder
2. In your `.env` file, define `MY_REMOTE_API_CERTIFICATE=/privatecredentials/my_remote_api_certificate.cer`
3. Have the HTTP client in your agent access the file using the ENV value



## Production Options

In production, you will need to make the file based credentials available to the application and use the environment variable to define where the container can access them.

There are a number of approaches:

- Use Kubernetes Secrets or Docker Secrets mounted as data volumes that will let your app see them as files on the filesystem
- Use the secrets management feature of your cloud hosting environment (Google Cloud Secrets, AWS Secrets Manager, etc)
- Use a 3rd party secrets management platform
- Manually place the credentials on your Docker hosts and mount volumes to map the credentials to the container (Less secure)
