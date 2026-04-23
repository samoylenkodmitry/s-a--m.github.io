---
layout: leetcode-entry
title: "1534. Count Good Triplets"
permalink: "/leetcode/problem/2025-04-14-1534-count-good-triplets/"
leetcode_ui: true
entry_slug: "2025-04-14-1534-count-good-triplets"
---

[1534. Count Good Triplets](https://leetcode.com/problems/count-good-triplets/description) easy
[blog post](https://leetcode.com/problems/count-good-triplets/solutions/6649234/kotlin-rust-by-samoylenkodmitry-lppy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14042025-1534-count-good-triplets?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/yJCbqHy4XjI)
![1.webp](/assets/leetcode_daily_images/cbad14bd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/958

#### Problem TLDR

Triplets with diff less than a,b,c #easy

#### Intuition

Brute force.

#### Approach

* I think it is possible to sort and do 3-pointers/binary search, but gave up

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countGoodTriplets(n: IntArray, a: Int, b: Int, c: Int) =
    n.indices.sumOf { i -> n.indices.sumOf { j -> n.indices.count { k ->
        i < j && j < k &&
        abs(n[i] - n[j]) <= a &&
        abs(n[j] - n[k]) <= b &&
        abs(n[i] - n[k]) <= c }}}

```
```rust

    pub fn count_good_triplets(n: Vec<i32>, a: i32, b: i32, c: i32) -> i32 {
        let mut r = 0;
        for i in (0..n.len()) { for j in (i + 1..n.len()) { for k in (j + 1..n.len()) {
            if (n[i] - n[j]).abs() <= a &&
               (n[j] - n[k]).abs() <= b &&
               (n[i] - n[k]).abs() <= c { r += 1 }
        }}} r
    }

```
```c++

    int countGoodTriplets(vector<int>& x, int a, int b, int c) {
        int n = size(x), r = 0;
        for (int i = 0; i < n - 2; ++i) for (int j = i + 1; j < n - 1; ++j) for (int k = j + 1; k < n; ++k)
        r += abs(x[i] - x[j]) <= a && abs(x[j] - x[k]) <= b && abs(x[i] - x[k]) <= c;
        return r;
    }

```

