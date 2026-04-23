---
layout: leetcode-entry
title: "3719. Longest Balanced Subarray I"
permalink: "/leetcode/problem/2026-02-10-3719-longest-balanced-subarray-i/"
leetcode_ui: true
entry_slug: "2026-02-10-3719-longest-balanced-subarray-i"
---

[3719. Longest Balanced Subarray I](https://leetcode.com/problems/longest-balanced-subarray-i/description/) medium
[blog post](https://leetcode.com/problems/longest-balanced-subarray-i/solutions/7568106/kotlin-rust-by-samoylenkodmitry-420q/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10022026-3719-longest-balanced-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/5Sr-e4d4sT0)

![a329e53f-9de6-4ee1-93ec-567f59924069 (1).webp](/assets/leetcode_daily_images/96fbff42.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1264

#### Problem TLDR

Longes even=odd subarray #medium #hashset

#### Intuition

```j
    // two pointers: we can't just blindly shrink/expand the window
    // pref mid suf
    //
    // 1 3 5 6 7 9 11
    // o o o e o o o
    // which part to cut?
    // is this DP? big acceptance rate, probably brain teaser
    // the array size is small only 1500
    // we can check every subarray in O(n^2)
    // build a prefix sum array
    // ok but how to deal with duplicates?
    //
    // 1 2 3 2
    // 1 1 2 2 odds
    // 0 1 1 2 evens non-uniq
    // 0 1 1 1 evens uniq
    //     * * will give wrong count for this range
    //
    // let's try write o(n^3), can't think of an optimal solution
    // TLE
    // as expected
    // 18 minute, go for hints; brute force? i litterally did that
    // ok maybe there is an O(N^2) brute-force possible?
```
From every position go to the end and count evens and odds. Mark visited with hashset.

#### Approach

* use array instead of hashset
* array can be global if we mark visited values with current i

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 263ms
    fun longestBalanced(n: IntArray) = n.indices.maxOf { i ->
        var b = 0; var m = HashSet<Int>()
        1 - i + ((i..<n.size).lastOrNull { j ->
            if (m.add(n[j])) b += 1 - n[j]%2*2; b == 0
        }?:i-1)
    }
```
```rust
// 16ms
    pub fn longest_balanced(n: Vec<i32>) -> i32 {
        let (mut r, mut s) = (0, [0;100001]);
        for i in 0..n.len() { if n.len() - i <= r { break }; let mut b = 0;
            for (j, &x) in n[i..].iter().enumerate() { let v = x as usize;
            if s[v] <= i { s[v] = i + 1; b += 1 - (v as i32 & 1) * 2 }
            if b == 0 { r = r.max(j + 1) }}} r as _
    }
```

