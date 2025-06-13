const fastify = require('fastify')();

fastify.get('/', (req, res) => {
        res.send('Hello World');
});

fastify.get('/users', (req, res) => {
    // TODO: Retrieve all users
});

fastify.get('/users/:id', (req, res) => {
    // TODO: Retrieve a specific user by ID
});

fastify.post('/users', (req, res) => {
    // TODO: Create a new user
});

fastify.put('/users/:id', (req, res) => {
    // TODO: Update a specific user by ID
});

fastify.delete('/users/:id', (req, res) => {
    // TODO: Delete a specific user by ID
});

const port = 3000;

fastify.listen(port, (err) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`Server is running on port ${port}`);
});
