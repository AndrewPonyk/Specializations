import React, { useEffect, useState } from 'react';
import './App.css';
import ProductList from './ProductList';
import axios from 'axios';
import CartList from './CartList';

function App() {
  const [products, setProducts] = useState([]);
  const [cart, setCart] = useState([]);

  useEffect(() => {
    axios.get('/products')
      .then(response => setProducts(response.data))
      .catch(error => console.log(error));

    axios.get('/cart')
      .then(response => setCart(response.data))
      .catch(error => console.log(error));
  }, []);

  const handleAddToCart = async (product) => {
    setCart([...cart, product]);
  }

  return (
    <div className="app-container">
      <h1 className="main-title">My Ecommerce Site</h1>
      <h3 className="section-title">Products</h3>
      <ProductList products={products} onAddToCart={handleAddToCart} />
      <h3 className="section-title">Cart</h3>
      <CartList products={cart} />
    </div>
  );
}

export default App;
