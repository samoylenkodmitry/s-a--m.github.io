---
layout: leetcode-entry
title: "2570. Merge Two 2D Arrays by Summing Values"
permalink: "/leetcode/problem/2025-03-02-2570-merge-two-2d-arrays-by-summing-values/"
leetcode_ui: true
entry_slug: "2025-03-02-2570-merge-two-2d-arrays-by-summing-values"
---

[2570. Merge Two 2D Arrays by Summing Values](https://leetcode.com/problems/merge-two-2d-arrays-by-summing-values/description/) easy
[blog post](https://leetcode.com/problems/merge-two-2d-arrays-by-summing-values/solutions/6484913/kotlin-rust-by-samoylenkodmitry-nv0o/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02032025-2570-merge-two-2d-arrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/jnA1rA0lpdk)
![1.webp](/assets/leetcode_daily_images/e4adec8d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/912

#### Problem TLDR

Merge two ascending [key, value] lists #easy

#### Intuition

The possibilities are:
* two pointers: increase the smallest
* map then sort
* sorted map
* use array as a map

#### Approach

* let's golf

#### Complexity

- Time complexity:
$$O(n)$$, or O(nlog(n)) for sorting

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun mergeArrays(a: Array<IntArray>, b: Array<IntArray>) = (a + b)
    .groupBy { it[0] }.toSortedMap().map { (k, v) -> listOf(k, v.sumBy { it[1] })}

```
```rust

    pub fn merge_arrays(mut a: Vec<Vec<i32>>, mut b: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut s = [0; 1001]; for x in a.into_iter().chain(b) { s[x[0] as usize] += x[1] }
        (1..1001).filter(|&i| s[i] > 0).map(|i| vec![i as i32, s[i]]).collect()
    }

```
```Rust(map)
    pub fn merge_arrays(mut a: Vec<Vec<i32>>, mut b: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        a.into_iter().chain(b).fold(BTreeMap::new(), |mut m, v| { *m.entry(v[0]).or_default() += v[1]; m })
        .into_iter().map(|(k, v)| vec![k, v]).collect()
    }
```
```Rust(pointers)
    pub fn merge_arrays(mut a: Vec<Vec<i32>>, mut b: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let (mut r, mut i, mut j) = (vec![], 0, 0);
        while i < a.len() || j < b.len() { r.push((
            if i == a.len() { j += 1; &b[j - 1] } else if j == b.len() { i += 1; &a[i - 1] }
            else if a[i][0] == b[j][0] { a[i][1] += b[j][1]; j += 1; i += 1; &a[i - 1] }
            else if a[i][0] < b[j][0] { i += 1; &a[i - 1] } else { j += 1; &b[j - 1] }
        ).clone())}; r
    }
```
```c++

    vector<vector<int>> mergeArrays(vector<vector<int>>& a, vector<vector<int>>& b) {
        map<int, int> m; for (auto x: {a, b}) for (auto& v: x) m[v[0]] += v[1];
        vector<vector<int>> r; for (auto [k, v]: m) r.push_back({k, v}); return r;
    }

```

