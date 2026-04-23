---
layout: leetcode-entry
title: "2657. Find the Prefix Common Array of Two Arrays"
permalink: "/leetcode/problem/2025-01-14-2657-find-the-prefix-common-array-of-two-arrays/"
leetcode_ui: true
entry_slug: "2025-01-14-2657-find-the-prefix-common-array-of-two-arrays"
---

[2657. Find the Prefix Common Array of Two Arrays](https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/description/) medium
[blog post](https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/solutions/6278083/kotlin-rust-by-samoylenkodmitry-rmm7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14012025-2657-find-the-prefix-common?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BXx2ubU_I4w)
[deep-dive](https://notebooklm.google.com/notebook/48e9802f-498b-4136-8afe-3483855b2dca/audio)
![1.webp](/assets/leetcode_daily_images/6d9797e5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/864

#### Problem TLDR

A[..i] and B[..i] intersections sizes #medium #counting

#### Intuition

The problem size is small, for 50 elements brute-force is accepted.

The optimal solution is to do a running counting of visited elements.

#### Approach

* brute-force is the shortest code
* we can do a 50-bitmask

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findThePrefixCommonArray(A: IntArray, B: IntArray) =
        List(A.size) { A.slice(0..it).intersect(B.slice(0..it)).size }

```
```rust

    pub fn find_the_prefix_common_array(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {
        let (mut f, mut c) = (0u64, 0);
        (0..a.len()).map(|i| {
            let (a, b) = (1 << a[i] as u64, 1 << b[i] as u64);
            c += (f & a > 0) as i32; f |= a;
            c += (f & b > 0) as i32; f |= b;  c
        }).collect()
    }

```
```c++

    vector<int> findThePrefixCommonArray(vector<int>& A, vector<int>& B) {
        vector<int> f(A.size() + 1), res(A.size()); int cnt = 0;
        for (int i = 0; i < A.size(); ++i)
            res[i] = (cnt += (++f[A[i]] > 1) + (++f[B[i]] > 1));
        return res;
    }

```

