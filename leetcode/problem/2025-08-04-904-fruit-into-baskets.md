---
layout: leetcode-entry
title: "904. Fruit Into Baskets"
permalink: "/leetcode/problem/2025-08-04-904-fruit-into-baskets/"
leetcode_ui: true
entry_slug: "2025-08-04-904-fruit-into-baskets"
---

[904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/description/) medium
[blog post](https://leetcode.com/problems/fruit-into-baskets/solutions/7042478/kotlin-rust-by-samoylenkodmitry-tdo0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/4082025-904-fruit-into-baskets?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/TrGPw_NpnjU)
![1.webp](/assets/leetcode_daily_images/ae7b8978.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1070

#### Problem TLDR

Max consequent two-types range #medium #counting

#### Intuition

Scan from left to right.
Count current type and previous.
On a third type drop the previous.

#### Approach

* how many extra variables we need?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 39ms
    fun totalFruit(f: IntArray): Int {
        var p = -1; var c = 0; var j = 0
        return f.withIndex().maxOf { (i, t) ->
            if (t != f[j]) {
                if (t != p) c = i - j
                p = f[j]; j = i
            }
            ++c
        }
    }

```

```rust

// 0ms
    pub fn total_fruit(f: Vec<i32>) -> i32 {
        let (mut p, mut k, mut j) = (-1, 0, 0);
        f.iter().enumerate().map(|(i, &t)| {
            if t != f[j] {
                if t != p { k = j }
                p = f[j]; j = i
            }
            i - k + 1
        }).max().unwrap() as _
    }

```
```c++

// 0ms
    int totalFruit(vector<int>& f) {
        int r = 0;
        for (int i = 0, j = 0, p = -1, k = 0; i < size(f); ++i) {
            if (f[i] != f[j]) {
                if (f[i] != p) k = j;
                p = f[j]; j = i;
            }
            r = max(r, i - k + 1);
        } return r;
    }

```
```python

// 77ms
    def totalFruit(self, f: List[int]) -> int:
        r = j = k = 0; p = -1
        for i, t in enumerate(f):
            if t != f[j]:
                if t != p: k = j
                p,j = f[j],i
            r = max(r, i - k + 1)
        return r

```

