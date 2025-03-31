# Contributing to Airport Operations Dashboard

Thank you for your interest in contributing to the Airport Operations Dashboard project! This document provides guidelines for contributions.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and update with your database credentials:
   ```bash
   cp .env.example .env
   ```
4. Initialize the database:
   ```bash
   python init_db.py
   python generate_sample_data.py
   ```
5. Run the server:
   ```bash
   python server.py
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Write or update tests if applicable
4. Ensure your code follows the project's style and conventions
5. Commit your changes with a clear commit message

## Data Sources

If you're enhancing the prediction model with real data sources:

1. Update the `data_sources.py` file to connect to real APIs
2. Document the new data source in README.md
3. Provide sample API responses in the tests

## Submitting a Pull Request

1. Push your changes to your fork
2. Submit a pull request to the main repository
3. Provide a clear description of the changes
4. Link any related issues

## Project Roadmap Ideas

Consider contributing to these areas:

1. Integrating real-world data sources to replace simulated data
2. Adding a frontend dashboard
3. Enhancing the ML model with additional algorithms 
4. Improving test coverage
5. Adding CI/CD pipeline configurations 