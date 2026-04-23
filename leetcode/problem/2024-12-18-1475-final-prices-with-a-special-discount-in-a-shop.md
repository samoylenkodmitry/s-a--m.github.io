---
layout: leetcode-entry
title: "1475. Final Prices With a Special Discount in a Shop"
permalink: "/leetcode/problem/2024-12-18-1475-final-prices-with-a-special-discount-in-a-shop/"
leetcode_ui: true
entry_slug: "2024-12-18-1475-final-prices-with-a-special-discount-in-a-shop"
---

[1475. Final Prices With a Special Discount in a Shop](https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/description/) easy
[blog post](https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/solutions/6159206/kotlin-rust-by-samoylenkodmitry-4lxr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18122024-1475-final-prices-with-a?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ERsVQvwFzKw)
[deep-dive](https://notebooklm.google.com/notebook/3d259d3c-1a1f-4651-a707-3946882a0232/audio)
![1.webp](/assets/leetcode_daily_images/666a691f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/836

#### Problem TLDR

Subtract next smaller value #easy #monotonic_stack

#### Intuition

Brute force works.
The next thing to try is a monotonic stack: iterate from the end, always keep values lower or equal than the current.

The big brain solution is to iterate forward: pop values lower than the current and adjust result at its index with the current value discount.

#### Approach

* let's implement all of them
* we can do it in-place if needed

#### Complexity

- Time complexity:
$$O(n^2)$$ or O(n)

- Space complexity:
$$O(n)$$ or O(1) for brute-force in-place

#### Code

```kotlin

    fun finalPrices(prices: IntArray) = IntArray(prices.size) { i ->
        prices[i] - (prices.slice(i + 1..<prices.size)
                       .firstOrNull { it <= prices[i] } ?: 0)
    }

```
```rust

    pub fn final_prices(prices: Vec<i32>) -> Vec<i32> {
        let (mut s, mut r) = (vec![], vec![0; prices.len()]);
        for i in (0..prices.len()).rev() {
            while s.last().map_or(false, |&x| x > prices[i]) { s.pop(); }
            r[i] = prices[i] - s.last().unwrap_or(&0);
            s.push(prices[i])
        }; r
    }

```
```c++

    vector<int> finalPrices(vector<int>& p) {
        vector<int> s;
        for (int i = 0; i < p.size(); ++i) {
            while (s.size() && p[s.back()] >= p[i]) {
                p[s.back()] -= p[i]; s.pop_back();
            }
            s.push_back(i);
        } return p;
    }

```

