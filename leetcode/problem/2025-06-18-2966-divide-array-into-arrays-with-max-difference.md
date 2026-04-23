---
layout: leetcode-entry
title: "2966. Divide Array Into Arrays With Max Difference"
permalink: "/leetcode/problem/2025-06-18-2966-divide-array-into-arrays-with-max-difference/"
leetcode_ui: true
entry_slug: "2025-06-18-2966-divide-array-into-arrays-with-max-difference"
---

[2966. Divide Array Into Arrays With Max Difference](https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/description/) medium
[blog post](https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/solutions/6856702/kotlin-rust-by-samoylenkodmitry-y596/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18062025-2966-divide-array-into-arrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/o99VuGqZAmw)
![1.webp](/assets/leetcode_daily_images/34470c27.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1023

#### Problem TLDR

List of k-narrow tripplets #medium

#### Intuition

Sort to minimize distance betwee siblings

#### Approach

* Kotlin has a `chunked`

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 200ms
    fun divideArray(n: IntArray, k: Int) = n.sorted().chunked(3)
        .takeIf { it.all { it[2] - it[0] <= k }} ?: listOf()

```
```kotlin

// 23ms
    fun divideArray(n: IntArray, k: Int): Array<IntArray> {
        val f = IntArray(100001); for (x in n) ++f[x]; var x = 0
        val r = Array(n.size / 3) { IntArray(3) }
        for (r in r) {
            while (f[x] < 1) ++x; --f[x]; r[0] = x; val m = k + x
            while (x < m && f[x] < 1) ++x; --f[x]; r[1] = x
            while (x < m && f[x] < 1) ++x; --f[x]; r[2] = x
            if (f[x] < 0) return emptyArray()
        }
        return r
    }

```
```rust

// 3ms
    pub fn divide_array(mut n: Vec<i32>, k: i32) -> Vec<Vec<i32>> {
        n.sort_unstable(); n.chunks(3)
        .map(|c| if c[2] - c[0] > k { None } else { Some(c.to_vec()) })
        .collect::<Option<_>>().unwrap_or_default()
    }

```
```c++

// 0ms
    vector<vector<int>> divideArray(vector<int>& n, int k) {
        vector<vector<int>> r; sort(begin(n), end(n));
        for (int i = 0; i < size(n); i += 3)
            if (n[i + 2] - n[i] > k) return {};
            else r.push_back({n[i], n[i + 1], n[i + 2]});
        return r;
    }

```

