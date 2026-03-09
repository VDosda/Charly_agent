from agent.bootstrap.runtime_factory import create_runtime


def main() -> None:
    runtime = create_runtime()

    print("Agent ready. Type 'exit' to quit.\n")

    session_id = "cli-session"
    user_id = "local-user"

    while True:
        user_input = input(">> ")

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        response = runtime.handle_message(
            user_id=user_id,
            session_id=session_id,
            message=user_input,
        )

        print(response)


if __name__ == "__main__":
    main()

