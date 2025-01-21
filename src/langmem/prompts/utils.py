def get_trajectory_clean(messages):
    response = []
    for m in messages:
        response.append(m.pretty_repr())
    return "\n".join(response)
