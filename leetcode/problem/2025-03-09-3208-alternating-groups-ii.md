---
layout: leetcode-entry
title: "3208. Alternating Groups II"
permalink: "/leetcode/problem/2025-03-09-3208-alternating-groups-ii/"
leetcode_ui: true
entry_slug: "2025-03-09-3208-alternating-groups-ii"
---

[3208. Alternating Groups II](https://leetcode.com/problems/alternating-groups-ii/description) medium
[blog post](https://leetcode.com/problems/alternating-groups-ii/solutions/6516544/kotlin-rust-by-samoylenkodmitry-7q9h/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09032025-3208-alternating-groups?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/k58d6C0reqI)
![1.webp](/assets/leetcode_daily_images/61920904.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/920

#### Problem TLDR

Count cycling alterating k-subarrays #medium #two_pointers

#### Intuition

`Two pointers` solution:
* the right pointer goes at k distance from the left
* if next not alterating, stop, and move left = right
* otherwise res++ and move both +1

The more simple solution is the `counting`:
* move a single pointer
* increase on alteratings or set back to 1
* everything >= k is good

#### Approach

* the cycle simpler handled with the current and next instead of previous
* use `xor` to look cooler
* in Rust slices are O(1) memory, concat the tail
* golf in Kotlin costs O(n) memory

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

    fun numberOfAlternatingGroups(c: IntArray, k: Int) = {
        var f = 0; (c.asList() + c.take(k)).zipWithNext()
        .count { (a, b) -> f *= a xor b; ++f >= k }}()

```
```rust

    pub fn number_of_alternating_groups(c: Vec<i32>, k: i32) -> i32 {
        let mut f = 0; [&c[..], &c[..k as usize]].concat().windows(2)
        .map(|w| { f *= w[0] ^ w[1]; f += 1; (f >= k) as i32 }).sum()
    }

```
```c++

    int numberOfAlternatingGroups(vector<int>& c, int k) {
        int r = 0, f = 0, n = size(c), i = 0;
        for (; i < n + k - 1;) f *= c[i % n] ^ c[(i++ + 1) % n], r += k <= ++f;
        return r;
    }

```

