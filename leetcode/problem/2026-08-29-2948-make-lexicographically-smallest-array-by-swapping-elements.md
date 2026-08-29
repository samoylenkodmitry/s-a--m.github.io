---
layout: leetcode-entry
title: "2948. Make Lexicographically Smallest Array by Swapping Elements"
permalink: "/leetcode/problem/2026-08-29-2948-make-lexicographically-smallest-array-by-swapping-elements/"
leetcode_ui: true
entry_slug: "2026-08-29-2948-make-lexicographically-smallest-array-by-swapping-elements"
---

[2948. Make Lexicographically Smallest Array by Swapping Elements](https://leetcode.com/problems/make-lexicographically-smallest-array-by-swapping-elements/solutions/8488805/kotlin-rust-by-samoylenkodmitry-hgb1/) medium
[substack](https://dmitriisamoilenko.substack.com/p/29082026-2948-make-lexicographically?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/7l58B7W7sjA)

https://dmitrysamoylenko.com/leetcode/

![29.08.2026.webp](/assets/leetcode_daily_images/29.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1466

#### Problem TLDR

Smallest array by swapping numbers with diff in 0..L

#### Intuition

Find groups then sort inside each. All numbers are reachable inside the group.

#### Approach

* use heap or just sort ad-hoc

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun lexicographicallySmallestArray(n: IntArray, l: Int) = run {
        val m = HashMap<Int, PriorityQueue<Int>>()
        var q = PriorityQueue<Int>(); var p = -l - 1
        for (x in n.sorted()) {
            if (x - p > l) q = PriorityQueue()
            q += x; m[x] = q; p = x
        }
        IntArray(n.size) { m[n[it]]!!.poll() }
    }
```
```rust
    pub fn lexicographically_smallest_array(n: Vec<i32>, l: i32) -> Vec<i32> {
        let (mut r, mut s) = (n.clone(), (0..n.len()).collect::<Vec<_>>());
        s.sort_by_key(|&i| n[i]);
        for g in s.chunk_by(|&a, &b| n[b] - n[a] <= l) {
            let mut p = g.to_vec(); p.sort();
            for (i, &j) in p.into_iter().zip(g) { r[i] = n[j] }
        } r
    }
```

