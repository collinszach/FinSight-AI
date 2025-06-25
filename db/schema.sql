CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    description TEXT,
    amount NUMERIC(12, 2),
    category TEXT,
    direction TEXT,
    source TEXT,
    filename TEXT,
    UNIQUE (date, description, amount, source, filename)
);
