# GitHub MCP Agent

The GitHub MCP Agent is a specialized agent that uses GitHub MCP (Model Context Protocol) tools for repository management and development workflows. It's built using LangGraph's `create_react_agent` for a clean, ReAct (Reasoning and Acting) pattern implementation.

**This agent is designed as a demo for agents that use MCP (Model Context Protocol) servers and tools.**

## Features

- Repository management (create, clone, browse)
- Issue management (create, list, update, close)
- Pull request management (create, review, merge)
- Branch management (create, switch, merge)
- File operations (read, write, search)
- Commit operations (create, view history)

## Configuration

To enable the GitHub MCP Agent, you need to configure the following environment variables:

### Required Settings

```bash
# GitHub Personal Access Token (required for GitHub MCP server)
# If not set, the GitHub MCP agent will have no tools
GITHUB_PAT=your_github_personal_access_token_here
```

### Optional Settings

```bash
# GitHub MCP server URL (defaults to https://api.github.com/mcp)
MCP_GITHUB_SERVER_URL=https://api.github.com/mcp
```

## GitHub Personal Access Token

To use the GitHub MCP Agent, you need a GitHub Personal Access Token (PAT) with appropriate permissions. The GitHub MCP server provides a wide variety of tools for repository management, and different tools require different scopes.

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with the following scopes:
   - `repo` (Full control of private repositories) - Enables tools like `create_issue`, `create_pull_request`, `get_file_contents`, `list_commits`
   - `read:org` (Read org and team membership) - Enables organization-related tools
   - `read:user` (Read user profile data) - Enables user profile tools
   - `user:email` (Access user email addresses) - Enables email-related functionality

**Note**: The specific tools available depend on your PAT scopes. With the recommended scopes above, you'll have access to most repository management tools including creating issues, pull requests, reading file contents, and listing commits.

## Usage

Once configured, the GitHub MCP Agent will be available as `github-mcp-agent` in the service.

### Example Prompts

Here are some example prompts you can use with the GitHub MCP Agent:

- **"Describe the JoshuaC215/agent-service-toolkit repository"** - Shows repository information and README content
- **"List the recent commits in this repository"** - Displays recent commit history
- **"What files are in the src directory?"** - Lists files in a specific directory
- **"Show me the README file"** - Displays repository README content
- **"Create a new issue titled 'Bug: Login not working' with the description 'The login form is not responding to user input'"** - Creates a new issue
- **"What are the open issues in this repository?"** - Lists open issues
- **"Create a pull request from feature-branch to main with the title 'Add new feature'"** - Creates a pull request
- **"Show me information about this repository"** - Displays repository details

## Agent Behavior

The GitHub MCP Agent is designed to:

- Provide helpful explanations of GitHub operations
- Follow best practices for commits, PRs, and issues
- Be cautious with destructive operations
- Respect repository permissions and access controls
- Provide context about what it's doing and why

## Troubleshooting

### Common Issues

1. **"MCP client not available"**: Check that `GITHUB_PAT` is set
2. **"Failed to load MCP tools"**: Verify your GitHub PAT has the required permissions
3. **Authentication errors**: Ensure your GitHub PAT is valid and not expired

### Logs

The service will log MCP-related information:
- `"MCP client initialized successfully"` - MCP client started
- `"Configured GitHub MCP agent with X tools"` - Tools loaded successfully
- `"GITHUB_PAT is not set, GitHub MCP agent will have no tools"` - GitHub PAT not configured

## Security Considerations

- Store your GitHub PAT securely (use environment variables or secure secret management)
- Regularly rotate your GitHub PAT
- Use minimal required permissions for your GitHub PAT
- Monitor GitHub API usage to avoid rate limits
