const express = require('express');
const cookieParser = require('cookie-parser');
const products = require('./product-test-data');

const app = express();

app.use(cookieParser());
app.use(express.json());

app.get('/products', (req, res) => {
  res.json(products);
});

app.post('/cart', (req, res) => {
  const productId = req.body.id;
  const cart = req.cookies.cart || [];
  
  cart.push(productId);
  
  res.cookie('cart', cart);
  res.sendStatus(200);
});

app.get('/cart', (req, res) => {
  const cart = req.cookies.cart || [];
  const cartProducts = cart.map(productId => {
    return products.find(product => product.id === productId);
  });
  
  res.json(cartProducts);
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
