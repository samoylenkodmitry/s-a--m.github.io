---
layout: leetcode-entry
title: "1346. Check If N and Its Double Exist"
permalink: "/leetcode/problem/2024-12-01-1346-check-if-n-and-its-double-exist/"
leetcode_ui: true
entry_slug: "2024-12-01-1346-check-if-n-and-its-double-exist"
---

[1346. Check If N and Its Double Exist](https://leetcode.com/problems/check-if-n-and-its-double-exist/description/) easy
[blog post](https://leetcode.com/problems/check-if-n-and-its-double-exist/solutions/6099818/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01122024-1346-check-if-n-and-its?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/o8OYYtC9vHk)
[deep-dive](https://notebooklm.google.com/notebook/90f29a3d-8434-4fbf-b052-7c2c0ab07f7c/audio)
![1.webp](/assets/leetcode_daily_images/de8d65b5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/818

#### Problem TLDR

Any `i != j && a[i] = 2 * a[j]` #easy

#### Intuition

Several ways:
* brute-force O(n^2) and O(1) memory
* HashSet / bitset O(n) and O(n) memory
* sort & binary search O(nlogn) and O(logn) memory
* bucket sort O(n) and O(n) memory

#### Approach

* corner case is `0`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun checkIfExist(arr: IntArray) = arr.groupBy { it }
        .run { keys.any { it != 0 && it * 2 in keys }
            || get(0)?.size ?: 0 > 1 }

```
```rust

    pub fn check_if_exist(mut arr: Vec<i32>) -> bool {
        arr.sort_unstable(); (0..arr.len()).any(|i| {
            i != arr.binary_search(&(2 * arr[i])).unwrap_or(i) })
    }

```
```c++

    bool checkIfExist(vector<int>& a) {
        int l = 1e3, f[2001] = {}; for (int x: a) ++f[x + l];
        for (int x = 500; --x;)
            if (f[l + x] && f[l + x * 2] || f[l - x] && f[l - x * 2])
                return 1;
        return f[l] > 1 ? 1 : 0;
    }

```
```c++

    bool checkIfExist(vector<int>& a) {
        int l = 2000; bitset<4001>b;
        for (int x: a) if (b[x * 2 + l] || x % 2 < 1 && b[x / 2 + l])
            return 1; else b[x + l] = 1;
        return 0;
    }

```

