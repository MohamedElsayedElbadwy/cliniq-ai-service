ClinIQ AI Service — AI Agent Instructions

1. Mission

You are working inside the shared ClinIQ AI Service repository.

Your responsibility is not only to implement the requested feature.

You MUST preserve:

* Existing features
* Existing API contracts
* Existing tests
* Project architecture
* Other developers’ work
* main branch stability

A feature is NOT considered complete if it works by itself but breaks another existing feature.

⸻

2. Non-Negotiable Rules

These rules MUST be followed for every task.

MUST

* Read this file before modifying code.
* Inspect the current repository state before coding.
* Work on a feature branch, never directly on main.
* Check the latest origin/main before starting feature work.
* Inspect existing architecture before adding new code.
* Keep new features modular.
* Run tests before creating a PR.
* Run the full test suite before declaring the task complete.
* Verify that existing features still work.
* Update documentation when API behavior or setup changes.
* Keep secrets out of the repository.
* Keep large model weights out of Git.

MUST NOT

* Modify main directly.
* Delete or disable existing functionality to make a new feature work.
* Rewrite another developer’s feature without explicit permission.
* Rename or change an existing API endpoint without approval.
* Change existing request/response schemas without checking compatibility.
* Remove tests because they are failing.
* Replace working architecture with a different architecture without approval.
* Add duplicate FastAPI applications or servers.
* Commit API keys, tokens, passwords, or .env files.
* Commit large model files.
* Force-push shared branches unless explicitly instructed.
* Reset another developer’s work.
* Mark a task as complete while existing functionality is broken.

⸻

3. Current Project Architecture

The project is a centralized FastAPI AI service.

Current structure:

cliniq-ai-service/
│
├── AGENTS.md
├── README.md
├── main.py
├── requirements.txt
├── .env.example
│
├── voice/
│   ├── router.py
│   ├── service.py
│   ├── config.py
│   └── schemas.py
│
├── mri/
│   ├── router.py
│   ├── service.py
│   └── ...
│
└── future AI modules/

The service currently contains AI capabilities such as:

* Real-Time Voice Assistant
* Brain MRI classification

Future AI capabilities may include:

* Patient RAG
* Doctor RAG
* Patient AI Assistant
* Doctor AI Assistant
* Medical Computer Vision
* Other AI/Agent features

Each feature SHOULD remain isolated inside its own module.

⸻

4. Feature Modularity

A new AI feature SHOULD follow this pattern when appropriate:

<feature>/
├── router.py
├── service.py
├── schemas.py
├── config.py
└── tests/

Do not force every feature to contain every file.

Use only the files actually required by the feature.

Example:

voice/
mri/
patient_rag/
doctor_rag/

Each feature should own its implementation.

Do not place feature-specific logic directly inside main.py.

⸻

5. FastAPI Architecture

main.py is a shared and sensitive file.

Its primary responsibilities are:

* Creating the FastAPI application
* Registering routers
* Configuring application startup/lifespan
* Serving shared application resources

A new feature SHOULD normally be integrated through a router.

Example:

from patient_rag.router import router as patient_rag_router
app.include_router(
    patient_rag_router,
    prefix="/api/ai/patient-rag",
)

Do NOT move large amounts of business logic into main.py.

Do NOT duplicate the FastAPI application.

Do NOT create another server unless explicitly required.

⸻

6. API Rules

All AI APIs SHOULD follow:

/api/ai/<feature>

Examples:

POST /api/ai/voice/turn
GET  /api/ai/mri/health
POST /api/ai/mri/predict

For new features:

/api/ai/<feature>/...

Existing API contracts are considered stable.

Do NOT silently:

* Rename endpoints
* Remove endpoints
* Rename request fields
* Rename response fields
* Change response formats
* Change HTTP methods

If a breaking API change is genuinely required:

STOP and ask for human approval.

⸻

7. Protect Existing Features

Before modifying shared code, identify which existing features may be affected.

At minimum, check:

Voice
MRI
Application startup
Swagger
Tests

A new feature must not make an existing feature unavailable.

For example:

If an optional AI model fails to load, it SHOULD NOT prevent unrelated features from starting unless that behavior is explicitly required.

Prefer isolated initialization for heavy or optional AI components.

⸻

8. Before Coding

Before writing code, inspect the repository.

Run:

git status
git branch --show-current
git fetch origin
git log --oneline --decorate -5

Then inspect the current architecture:

dir

On PowerShell, inspect important files as needed:

Get-Content main.py
Get-Content requirements.txt

Also inspect the relevant existing feature before changing shared code.

The goal is to understand the current state before making assumptions.

⸻

9. Working With main

Before creating or updating a feature branch, make sure the feature is based on the latest main.

Typical workflow:

git fetch origin
git status
git branch --show-current

If you are starting new work:

git switch main
git pull --ff-only origin main
git switch -c feature/<feature-name>

If you are already working on a feature branch:

git fetch origin
git merge origin/main

Resolve conflicts carefully.

Do NOT blindly overwrite files during conflict resolution.

After resolving conflicts:

git status

Then run the tests again.

⸻

10. Shared Files

Treat these files as high-risk:

main.py
requirements.txt
README.md
.github/*
Docker/*
shared/*

Changes to these files must be minimal and intentional.

If a feature can be implemented without changing a shared file, prefer that approach.

If changing a shared file could affect another feature:

STOP and evaluate the impact before proceeding.

⸻

11. Dependencies

Before adding a dependency:

1. Check whether an existing dependency already provides the required functionality.
2. Check whether the dependency conflicts with existing packages.
3. Keep the dependency necessary and justified.
4. Run the test suite after installation.

Do NOT upgrade unrelated dependencies just because newer versions exist.

Do NOT remove existing dependencies without checking their usage.

If a dependency upgrade may affect TensorFlow, Keras, Groq, FastAPI, or another critical component:

STOP and verify compatibility first.

⸻

12. Environment Variables and Secrets

Never commit:

.env
API keys
Tokens
Passwords
Credentials
Private secrets

Use:

.env.example

for documenting required variables.

Example:

GROQ_API_KEY=

Code must read secrets from environment variables.

Never hard-code credentials.

Never print API keys or secrets in logs.

⸻

13. Model Files

Large model weights must NOT be committed to Git unless explicitly approved.

Use appropriate local/model storage mechanisms instead.

Examples:

*.h5
*.keras
*.pt
*.pth
*.onnx

If a model file is required for local execution, document how developers obtain it.

⸻

14. Testing Requirements

Every new feature SHOULD include tests.

Before creating a PR, run:

python -m pytest -v

The full test suite must pass.

Do not run only the tests belonging to the new feature and assume everything is fine.

The minimum integration check is:

New Feature     ✅
Voice           ✅
MRI             ✅
All Tests       ✅
App Startup     ✅
Swagger         ✅
No Breaking API ✅

⸻

15. Application Startup Check

After significant changes, verify that the application starts:

python -m uvicorn main:app --reload --port 8080

Verify:

http://127.0.0.1:8080
http://127.0.0.1:8080/docs

The application should start without breaking unrelated features.

⸻

16. Manual API Verification

For a new endpoint:

1. Start FastAPI.
2. Open Swagger.
3. Verify the endpoint appears.
4. Send a valid request.
5. Send an invalid request if applicable.
6. Verify the response schema.
7. Verify existing endpoints still work.

Swagger:

/docs

⸻

17. External Services

If a feature uses an external AI provider or API:

* Keep credentials in environment variables.
* Handle provider errors safely.
* Avoid exposing provider secrets.
* Add tests that do not require live API calls whenever possible.
* Use mocks for unit tests when appropriate.
* Document required environment variables.

Live API testing can be performed separately when credentials are available.

⸻

18. Error Handling

Errors should be handled intentionally.

Do NOT:

except Exception:
    pass

Do NOT hide errors just to make tests pass.

Errors should provide useful information while avoiding sensitive data.

External service failures should not silently corrupt application state.

⸻

19. Documentation

Update README.md when a feature changes:

* API endpoints
* Setup requirements
* Environment variables
* Running instructions
* Testing instructions
* Feature availability

Documentation should describe the actual implementation.

Do not document features that do not exist.

⸻

20. Git Commit Rules

Use clear commits.

Examples:

feat: add patient rag endpoint
fix: handle voice transcription failure
test: add patient rag tests
docs: update patient rag setup
refactor: isolate mri model loading

Avoid meaningless commits such as:

update
fix
changes
test
final

Keep commits focused when possible.

⸻

21. Pull Request Rules

Create one PR for one feature or logically related change.

A PR should explain:

## Summary
What was added?
## Changes
Which files/modules changed?
## API
What endpoints were added or changed?
## Tests
What tests were run?
## Compatibility
Were existing features affected?
## Manual Verification
What was tested manually?
## Risks
Are there any known risks?

Do NOT merge your own PR unless the repository policy explicitly allows it.

The PR should be reviewed before merging into main.

⸻

22. Before Creating a PR

Always synchronize with the latest main.

Typical workflow:

git fetch origin
git merge origin/main

Resolve conflicts if necessary.

Then:

python -m pytest -v

Then inspect:

git status
git diff origin/main...HEAD

Verify that the diff contains only intentional changes.

Then push:

git push origin <your-branch>

Create/update the PR.

⸻

23. Never Hide Integration Problems

If your feature only works because:

* Another feature was removed
* A test was deleted
* An endpoint was changed
* A dependency was downgraded unexpectedly
* Existing code was overwritten
* A shared file was modified incorrectly

then the feature is NOT complete.

Fix the integration problem instead.

⸻

24. Stop and Ask for Human Approval

The agent MUST stop and ask before doing any of the following:

* Breaking an existing API
* Removing an existing feature
* Changing the overall architecture
* Replacing the modular monolith architecture
* Introducing a major new infrastructure component
* Changing database architecture
* Changing authentication architecture
* Changing CI/CD rules
* Changing branch protection
* Deleting another developer’s work
* Force-pushing a shared branch
* Moving large parts of the repository
* Making a major dependency change
* Changing another feature’s behavior
* Making changes that cannot be confidently verified

When uncertain:

Do not guess. Ask.

⸻

25. Conflict Resolution Rules

When Git reports a conflict:

1. Understand both sides.
2. Identify which changes belong to main.
3. Identify which changes belong to the feature branch.
4. Preserve valid work from both sides.
5. Do not automatically choose “ours” or “theirs”.
6. Run tests after resolving the conflict.

For conflicts in main.py:

Pay special attention to router imports and:

app.include_router(...)

Do not accidentally remove an existing feature during conflict resolution.

⸻

26. Definition of Done

A feature is DONE only when all applicable items are true:

[ ] Feature implemented
[ ] Feature is modular
[ ] Existing features preserved
[ ] API contract verified
[ ] Environment variables documented
[ ] No secrets committed
[ ] Tests added/updated
[ ] Full test suite passes
[ ] Application starts successfully
[ ] Swagger verified
[ ] Existing endpoints checked
[ ] README updated if necessary
[ ] Latest main integrated
[ ] Git diff reviewed
[ ] PR created

If one of the critical items fails:

The task is not complete.

⸻

27. Golden Rule

Integrate features — don’t just merge them.

The goal of this repository is to allow multiple developers and AI agents to build different AI capabilities without breaking the existing ClinIQ system.

Every change must leave the repository in a stable and working state.
