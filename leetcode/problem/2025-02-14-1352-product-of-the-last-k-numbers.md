---
layout: leetcode-entry
title: "1352. Product of the Last K Numbers"
permalink: "/leetcode/problem/2025-02-14-1352-product-of-the-last-k-numbers/"
leetcode_ui: true
entry_slug: "2025-02-14-1352-product-of-the-last-k-numbers"
---

[1352. Product of the Last K Numbers](https://leetcode.com/problems/product-of-the-last-k-numbers/description/) medium
[blog post](https://leetcode.com/problems/product-of-the-last-k-numbers/solutions/6421014/kotlin-rust-by-samoylenkodmitry-eknq/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14022025-1352-product-of-the-last?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/tRJ5jmTCAL0)
![1.webp](/assets/leetcode_daily_images/ccb54926.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/895

#### Problem TLDR

Running suffix product #medium #math #prefix_product

#### Intuition

The brute-force is accepted.

Didn't found myself O(1) solution, just wasn't prepared to the math fact: prefix product can work for positive numbers.

#### Approach

* edge case is `1` for the initial product

#### Complexity

- Time complexity:
$$O(n^2)$$ for brute-force, O(n) for prefix-product

- Space complexity:
$$O(n)$$

#### Code

```kotlin

class ProductOfNumbers(): ArrayList<Int>() {
    fun getProduct(k: Int) = takeLast(k).reduce(Int::times)
}

```
```rust

struct ProductOfNumbers(usize, Vec<i32>);
impl ProductOfNumbers {
    fn new() -> Self { Self(0, vec![1]) }
    fn add(&mut self, n: i32) {
        if n > 0 { self.1.push(n * self.1[self.0]); self.0 += 1 }
        else { self.0 = 0; self.1.resize(1, 0) }
    }
    fn get_product(&self, k: i32) -> i32 {
        if k as usize > self.0 { 0 } else { self.1[self.0] / self.1[self.0 - k as usize] }
    }
}

```
```c++

class ProductOfNumbers {
    int c = 0; vector<int> p = {1};
public:
    void add(int n) { n > 0 ? (p.push_back(n * p.back()), ++c) : (p.resize(1, 0), c = 0); }
    int getProduct(int k) { return k > c ? 0 : p[c] / p[c - k]; }
};

```

