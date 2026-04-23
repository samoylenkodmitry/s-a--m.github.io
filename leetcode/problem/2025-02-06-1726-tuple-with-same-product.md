---
layout: leetcode-entry
title: "1726. Tuple with Same Product"
permalink: "/leetcode/problem/2025-02-06-1726-tuple-with-same-product/"
leetcode_ui: true
entry_slug: "2025-02-06-1726-tuple-with-same-product"
---

[1726. Tuple with Same Product](https://leetcode.com/problems/tuple-with-same-product/description/) medium
[blog post](https://leetcode.com/problems/tuple-with-same-product/solutions/6383872/kotlin-rust-by-samoylenkodmitry-e311/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06022025-1726-tuple-with-same-product?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RUqXLlIwEKs)
![webp](/assets/06_02_25.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/887

#### Problem TLDR

Count uniq 4-tupples a*b=c*d #medium #counting #sort

#### Intuition

We should count frequencies of the result a[i] * a[j].
For every tupple a * b == c * d, we have total 8 permutations:
a b = c d
a b = d c
b a = c d
b a = d c
c d = a b
c d = b a
d c = a b
d c = b a.

How to count them in a single pass? Let's count only `uniq` pairs and multiply them by 8:

```j

    // 2 3 4 6
    // 2*3 2*4 2*6
    //   3*4 3*6
    //     4*6

```

#### Approach

* We can avoid the HashMap by storing all results in a list, then sorting it and walk linearly. Number of permutations depends on the frequency: p = f * (f - 1) / 2.

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

    fun tupleSameProduct(nums: IntArray): Int {
        val f = HashMap<Int, Int>(); var res = 0
        for (i in nums.indices) for (j in i + 1..<nums.size) {
            val ab = nums[i] * nums[j]; res += 8 * (f[ab] ?: 0); f[ab] = 1 + (f[ab] ?: 0)
        }
        return res
    }

```
```rust

    pub fn tuple_same_product(nums: Vec<i32>) -> i32 {
        let mut f = vec![];
        for i in 0..nums.len() { for j in i + 1..nums.len() { f.push(nums[i] * nums[j]) }};
        f.sort_unstable();
        f.chunk_by(|a, b| a == b).map(|c| 4 * c.len() * (c.len() - 1)).sum::<usize>() as i32
    }

```
```c++

    int tupleSameProduct(vector<int>& n) {
        int r = 0, m = size(n); unordered_map<int, int> f;
        for (int i = 0; i < m; ++i)
            for (int j = i + 1; j < m; r += f[n[i] * n[j++]]++);
        return r * 8;
    }

```

