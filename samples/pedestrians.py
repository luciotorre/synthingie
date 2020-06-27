import synthingie as st


def pedestrian():
    wave = st.Sin(2500, 0.2)
    gate = st.NaiveSquare(5, amplitude=0.5) + 0.5

    pedestrian = wave * gate

    st.play(pedestrian)


if __name__ == "__main__":
    pedestrian()
