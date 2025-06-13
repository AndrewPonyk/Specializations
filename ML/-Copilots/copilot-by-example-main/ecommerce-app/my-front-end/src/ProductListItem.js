import React from 'react';
import axios from 'axios';
import './ProductListItem.css';

function ProductListItem({ product, onAddToCart }) {
  const { id, name, price, description, imageUrl } = product;

  const addToCart = async () => {
    await axios.post('/cart', { id });
    alert('Item added to cart');
    onAddToCart(product);
  };

  return (
    <div className="product-list-item">
      {imageUrl && (
        <div className="product-image-container">
          <img src={imageUrl} alt={name} className="product-image" />
        </div>
      )}
      <h2>{name}</h2>
      <p>{description}</p>
      <p className="price">${price}</p>
      <button onClick={addToCart}>Add to Cart</button>
    </div>
  );
}

export default ProductListItem;
