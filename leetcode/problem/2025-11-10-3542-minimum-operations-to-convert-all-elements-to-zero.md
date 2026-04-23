---
layout: leetcode-entry
title: "3542. Minimum Operations to Convert All Elements to Zero"
permalink: "/leetcode/problem/2025-11-10-3542-minimum-operations-to-convert-all-elements-to-zero/"
leetcode_ui: true
entry_slug: "2025-11-10-3542-minimum-operations-to-convert-all-elements-to-zero"
---

[3542. Minimum Operations to Convert All Elements to Zero](https://leetcode.com/problems/minimum-operations-to-convert-all-elements-to-zero/description) medium
[blog post](https://leetcode.com/problems/minimum-operations-to-convert-all-elements-to-zero/solutions/7338992/kotlin-rust-by-samoylenkodmitry-6yuz/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10112025-3542-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Xz3zCI04SM0)

![1827673c-dd69-45f1-b40c-66834545a841 (1).webp](/assets/leetcode_daily_images/017b326b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1169

#### Problem TLDR

Min ops to zeros by making min of subarrays 0 #medium #monotonic_stack

#### Intuition

```j
    // 1 2 3 1 2 3 4 1 2 3
    // 0 2 3 0 2 3 4 0 2 3
    // 0 0 3 0 2 3 4 0 2 3
    // 0 0 0 0 2 3 4 0 2 3
    // 0 0 0 0 0 3 4 0 2 3
    // 0 0 0 0 0 0 4 0 2 3
    // 0 0 0 0 0 0 0 0 2 3
    // 0 0 0 0 0 0 0 0 0 3
    // 0 0 0 0 0 0 0 0 0 0
    //
    // 1 2 3 2 1
    // 0 2 3 2 0
    // 0 0 3 0 0
    // 0 0 0 0 0
    //
    // 3 2 1 2 3
    // 3 2 0 2 3
    // 3 0 0 2 3
    // 0 0 0 2 3
    // 0 0 0 0 3
    // 0 0 0 0 0
    //
    // 3 2 1 2 3 2 1
    // *               3
    //   *             2<3, +ops
    //     *           1<2, +ops
    //       2         2 1
    //         3       3 2 1
    //           2     2<3, +ops; 2 1
    //             1   1<2, +ops; 1
    //                 1 + ops
    //    total = 5
```

* each decrease means the previous value goes to the separate operation

#### Approach

* the remainig values in stack are all separate operations
* or, we can count *increases* as operations instead of *decreases*
* dirty trick to O(1) memory: use input array as a stack

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 32ms
    fun minOperations(n: IntArray): Int {
        var s = -1
        return n.count { x ->
            while (s >= 0 && n[s] > x) s--
            (x > 0 && (s < 0 || n[s] < x)).also { if (it) n[++s] = x}
        }
    }
```
```rust
// 12ms
    pub fn min_operations(n: Vec<i32>) -> i32 {
        let (mut r, mut s) = (0, vec![0]);
        for x in n {
            while s[s.len()-1] > x { s.pop(); }
            if s[s.len()-1] < x { r += 1; s.push(x) }
        }; r
    }
```

