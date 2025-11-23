-- ============================================================================
-- SQL Learning Agent - Sample Database with Parent-Child Relationships
-- ============================================================================

-- Drop tables if exist (for clean restart)
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS departments CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS addresses CASCADE;

-- ============================================================================
-- DEPARTMENTS & EMPLOYEES (Parent-Child: Department -> Employees)
-- ============================================================================

CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    department_id INTEGER REFERENCES departments(id),
    manager_id INTEGER REFERENCES employees(id),  -- Self-referential (Employee -> Manager)
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    salary DECIMAL(10, 2),
    hire_date DATE,
    is_active BOOLEAN DEFAULT true
);

-- ============================================================================
-- CATEGORIES & PRODUCTS (Parent-Child: Category -> Products)
-- ============================================================================

CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    parent_id INTEGER REFERENCES categories(id),  -- Self-referential (subcategories)
    name VARCHAR(100) NOT NULL,
    description TEXT
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    is_available BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CUSTOMERS & ADDRESSES (Parent-Child: Customer -> Addresses)
-- ============================================================================

CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE addresses (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id) ON DELETE CASCADE,
    address_type VARCHAR(20) DEFAULT 'shipping',  -- 'shipping' or 'billing'
    street VARCHAR(200),
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    country VARCHAR(50) DEFAULT 'USA',
    is_default BOOLEAN DEFAULT false
);

-- ============================================================================
-- ORDERS & ORDER_ITEMS (Parent-Child: Order -> OrderItems)
-- ============================================================================

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, processing, shipped, delivered, cancelled
    total_amount DECIMAL(12, 2),
    shipping_address_id INTEGER REFERENCES addresses(id)
);

CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) GENERATED ALWAYS AS (quantity * unit_price) STORED
);

-- ============================================================================
-- INSERT SAMPLE DATA
-- ============================================================================

-- Departments
INSERT INTO departments (name, budget) VALUES
    ('Engineering', 500000.00),
    ('Sales', 300000.00),
    ('Marketing', 200000.00),
    ('Human Resources', 150000.00),
    ('Finance', 250000.00);

-- Employees (with managers - hierarchical)
INSERT INTO employees (department_id, manager_id, first_name, last_name, email, salary, hire_date) VALUES
    (1, NULL, 'John', 'Smith', 'john.smith@company.com', 150000, '2020-01-15'),
    (1, 1, 'Alice', 'Johnson', 'alice.johnson@company.com', 120000, '2020-06-01'),
    (1, 1, 'Bob', 'Williams', 'bob.williams@company.com', 110000, '2021-03-15'),
    (1, 2, 'Carol', 'Brown', 'carol.brown@company.com', 95000, '2021-08-01'),
    (1, 2, 'David', 'Davis', 'david.davis@company.com', 90000, '2022-01-10'),
    (2, NULL, 'Emma', 'Wilson', 'emma.wilson@company.com', 130000, '2019-05-20'),
    (2, 6, 'Frank', 'Miller', 'frank.miller@company.com', 85000, '2021-02-01'),
    (2, 6, 'Grace', 'Taylor', 'grace.taylor@company.com', 80000, '2021-09-15'),
    (3, NULL, 'Henry', 'Anderson', 'henry.anderson@company.com', 125000, '2020-03-01'),
    (3, 9, 'Ivy', 'Thomas', 'ivy.thomas@company.com', 75000, '2022-04-01'),
    (4, NULL, 'Jack', 'Jackson', 'jack.jackson@company.com', 100000, '2019-08-15'),
    (5, NULL, 'Kate', 'White', 'kate.white@company.com', 140000, '2018-11-01');

-- Categories (with subcategories - hierarchical)
INSERT INTO categories (parent_id, name, description) VALUES
    (NULL, 'Electronics', 'Electronic devices and accessories'),
    (NULL, 'Clothing', 'Apparel and fashion items'),
    (NULL, 'Home & Garden', 'Home improvement and garden supplies'),
    (1, 'Computers', 'Desktop and laptop computers'),
    (1, 'Smartphones', 'Mobile phones and accessories'),
    (1, 'Audio', 'Headphones, speakers, and audio equipment'),
    (2, 'Men', 'Men clothing'),
    (2, 'Women', 'Women clothing'),
    (2, 'Kids', 'Children clothing'),
    (3, 'Furniture', 'Home furniture'),
    (3, 'Garden Tools', 'Outdoor and garden equipment'),
    (4, 'Laptops', 'Portable computers'),
    (4, 'Desktops', 'Desktop computers'),
    (5, 'iPhone', 'Apple smartphones'),
    (5, 'Android', 'Android smartphones');

-- Products
INSERT INTO products (category_id, name, description, price, stock_quantity) VALUES
    (12, 'MacBook Pro 14"', 'Apple MacBook Pro with M3 chip', 1999.99, 50),
    (12, 'Dell XPS 15', 'Dell premium laptop', 1599.99, 35),
    (12, 'ThinkPad X1 Carbon', 'Lenovo business laptop', 1449.99, 40),
    (13, 'iMac 24"', 'Apple all-in-one desktop', 1299.99, 25),
    (14, 'iPhone 15 Pro', 'Latest Apple smartphone', 999.99, 100),
    (14, 'iPhone 15', 'Apple smartphone', 799.99, 150),
    (15, 'Samsung Galaxy S24', 'Samsung flagship phone', 899.99, 80),
    (15, 'Google Pixel 8', 'Google smartphone', 699.99, 60),
    (6, 'AirPods Pro', 'Apple wireless earbuds', 249.99, 200),
    (6, 'Sony WH-1000XM5', 'Premium noise-cancelling headphones', 349.99, 75),
    (7, 'Classic Polo Shirt', 'Cotton polo shirt for men', 49.99, 300),
    (7, 'Slim Fit Jeans', 'Modern slim fit denim', 79.99, 250),
    (8, 'Summer Dress', 'Floral summer dress', 89.99, 150),
    (8, 'Yoga Pants', 'Comfortable yoga pants', 59.99, 200),
    (10, 'Modern Sofa', '3-seater fabric sofa', 899.99, 20),
    (10, 'Dining Table Set', '6-person dining set', 1299.99, 15),
    (11, 'Lawn Mower', 'Electric lawn mower', 399.99, 30),
    (11, 'Garden Hose 50ft', 'Heavy duty garden hose', 39.99, 100);

-- Customers
INSERT INTO customers (first_name, last_name, email, phone) VALUES
    ('Michael', 'Scott', 'michael.scott@email.com', '555-0101'),
    ('Dwight', 'Schrute', 'dwight.schrute@email.com', '555-0102'),
    ('Jim', 'Halpert', 'jim.halpert@email.com', '555-0103'),
    ('Pam', 'Beesly', 'pam.beesly@email.com', '555-0104'),
    ('Angela', 'Martin', 'angela.martin@email.com', '555-0105'),
    ('Kevin', 'Malone', 'kevin.malone@email.com', '555-0106'),
    ('Oscar', 'Martinez', 'oscar.martinez@email.com', '555-0107'),
    ('Stanley', 'Hudson', 'stanley.hudson@email.com', '555-0108'),
    ('Phyllis', 'Vance', 'phyllis.vance@email.com', '555-0109'),
    ('Andy', 'Bernard', 'andy.bernard@email.com', '555-0110');

-- Addresses
INSERT INTO addresses (customer_id, address_type, street, city, state, zip_code, is_default) VALUES
    (1, 'shipping', '123 Main St', 'Scranton', 'PA', '18503', true),
    (1, 'billing', '456 Office Blvd', 'Scranton', 'PA', '18503', false),
    (2, 'shipping', '789 Beet Farm Rd', 'Scranton', 'PA', '18504', true),
    (3, 'shipping', '321 Stamford Ave', 'Stamford', 'CT', '06901', true),
    (4, 'shipping', '654 Art District', 'Scranton', 'PA', '18505', true),
    (5, 'shipping', '987 Cat Lane', 'Scranton', 'PA', '18506', true),
    (6, 'shipping', '111 Chili Way', 'Scranton', 'PA', '18507', true),
    (7, 'shipping', '222 Finance St', 'Scranton', 'PA', '18508', true),
    (8, 'shipping', '333 Crossword Blvd', 'Scranton', 'PA', '18509', true),
    (9, 'shipping', '444 Vance Refrigeration', 'Scranton', 'PA', '18510', true),
    (10, 'shipping', '555 Cornell Dr', 'Scranton', 'PA', '18511', true);

-- Orders
INSERT INTO orders (customer_id, status, total_amount, shipping_address_id) VALUES
    (1, 'delivered', 2249.98, 1),
    (1, 'shipped', 349.99, 1),
    (2, 'delivered', 1999.99, 3),
    (3, 'processing', 899.99, 4),
    (4, 'pending', 149.98, 5),
    (5, 'delivered', 79.99, 6),
    (6, 'cancelled', 699.99, 7),
    (7, 'delivered', 1599.99, 8),
    (8, 'shipped', 439.98, 9),
    (9, 'pending', 1299.99, 10),
    (10, 'delivered', 999.99, 11),
    (1, 'processing', 2999.98, 1),
    (3, 'delivered', 179.98, 4);

-- Order Items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 1999.99),
    (1, 9, 1, 249.99),
    (2, 10, 1, 349.99),
    (3, 1, 1, 1999.99),
    (4, 7, 1, 899.99),
    (5, 11, 2, 49.99),
    (5, 12, 1, 79.99),
    (6, 12, 1, 79.99),
    (7, 8, 1, 699.99),
    (8, 2, 1, 1599.99),
    (9, 17, 1, 399.99),
    (9, 18, 1, 39.99),
    (10, 16, 1, 1299.99),
    (11, 5, 1, 999.99),
    (12, 1, 1, 1999.99),
    (12, 5, 1, 999.99),
    (13, 11, 2, 49.99),
    (13, 14, 1, 59.99);

-- ============================================================================
-- USEFUL VIEWS
-- ============================================================================

CREATE VIEW v_employee_hierarchy AS
SELECT
    e.id,
    e.first_name || ' ' || e.last_name AS employee_name,
    d.name AS department,
    m.first_name || ' ' || m.last_name AS manager_name,
    e.salary
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id
LEFT JOIN employees m ON e.manager_id = m.id;

CREATE VIEW v_order_summary AS
SELECT
    o.id AS order_id,
    c.first_name || ' ' || c.last_name AS customer_name,
    o.order_date,
    o.status,
    COUNT(oi.id) AS item_count,
    SUM(oi.subtotal) AS calculated_total
FROM orders o
JOIN customers c ON o.customer_id = c.id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.id, c.first_name, c.last_name, o.order_date, o.status;

CREATE VIEW v_category_tree AS
WITH RECURSIVE cat_tree AS (
    SELECT id, name, parent_id, name::text AS path, 1 AS level
    FROM categories
    WHERE parent_id IS NULL
    UNION ALL
    SELECT c.id, c.name, c.parent_id, ct.path || ' > ' || c.name, ct.level + 1
    FROM categories c
    JOIN cat_tree ct ON c.parent_id = ct.id
)
SELECT * FROM cat_tree ORDER BY path;

-- ============================================================================
-- Grant permissions
-- ============================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agent;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agent;
