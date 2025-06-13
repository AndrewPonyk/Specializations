import React from 'react';
import ProductListItem from './ProductListItem';

const ProductList = ({ products, onAddToCart }) => {
  return (
    <div>
      {products.map((product) => (
        <ProductListItem key={product.id} onAddToCart={onAddToCart} product={product} />
      ))}
    </div>
  );
};

export default ProductList;
