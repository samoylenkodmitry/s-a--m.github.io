---
layout: leetcode-entry
title: "3394. Check if Grid can be Cut into Sections"
permalink: "/leetcode/problem/2025-03-25-3394-check-if-grid-can-be-cut-into-sections/"
leetcode_ui: true
entry_slug: "2025-03-25-3394-check-if-grid-can-be-cut-into-sections"
---

[3394. Check if Grid can be Cut into Sections](https://leetcode.com/problems/check-if-grid-can-be-cut-into-sections/description/) medium
[blog post](https://leetcode.com/problems/check-if-grid-can-be-cut-into-sections/solutions/6576937/kotlin-rust-by-samoylenkodmitry-g9o6/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25032025-3394-check-if-grid-can-be?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_2BO5y2dn5k)
![1.webp](/assets/leetcode_daily_images/c09993ad.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/938

#### Problem TLDR

3 non-intersecting ranges on x or y axis #medium #line_sweep

#### Intuition

Solve problem on x axis then on y axis.
Several ways:
* sorting: sort by the start, check border, set border at max of the ends
* TreeMap: save starting as +1 and end points as -1, do line sweep and keep counter, 0 is a cut place
* heap: same as previous, but put pairs of coordinate and diff

#### Approach

* try to reuse the logic
* Rust uses ipnsort https://github.com/Voultapher/sort-research-rs/blob/main/writeup/ipnsort_introduction/text.md
* Kotlin uses Java (21) dual-pivot quicksort https://github.com/Voultapher/sort-research-rs/blob/main/writeup/ipnsort_introduction/text.md

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(log(n))$$ or O(1) if we implement O(1) sorting algorithm ourselves

#### Code

```kotlin

    fun checkValidCuts(n: Int, rec: Array<IntArray>) = (0..1).any { j ->
        var b = -1; rec.sortBy { it[j] }
        rec.count { r -> r[j] >= b.also { b = max(b, r[j + 2]) }} > 2
    }

```
```rust

    pub fn check_valid_cuts(n: i32, mut rec: Vec<Vec<i32>>) -> bool {
        (0..2).any(|j| {
            let mut b = -1; rec.sort_unstable_by_key(|r| r[j]);
            rec.iter().filter(|r| {
                let x = r[j] >= b; b = b.max(r[j + 2]); x }).count() > 2 })
    }

```
```c++

    bool checkValidCuts(int n, vector<vector<int>>& rec) {
        for (int j: {0, 1}) {
            sort(begin(rec), end(rec), [j](auto& a, auto& b) { return a[j] < b[j]; });
            int b = -1, c = 0; for (auto& r: rec) c += r[j] >= b, b = max(b, r[j + 2]);
            if (c > 2) return 1;
        } return 0;
    }

```

