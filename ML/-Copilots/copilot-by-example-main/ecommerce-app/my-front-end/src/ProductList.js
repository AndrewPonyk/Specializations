import React from 'react';
import ProductListItem from './ProductListItem';
import './ProductList.css';

const ProductList = ({ products, onAddToCart }) => {
  return (
    <div className="product-list-container">
      {products.map((product) => (
        <ProductListItem key={product.id} onAddToCart={onAddToCart} product={product} />
      ))}
    </div>
  );
};

export default ProductList;
