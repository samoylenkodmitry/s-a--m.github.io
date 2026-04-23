---
layout: leetcode-entry
title: "769. Max Chunks To Make Sorted"
permalink: "/leetcode/problem/2024-12-19-769-max-chunks-to-make-sorted/"
leetcode_ui: true
entry_slug: "2024-12-19-769-max-chunks-to-make-sorted"
---

[769. Max Chunks To Make Sorted](https://leetcode.com/problems/max-chunks-to-make-sorted/description/) medium
[blog post](https://leetcode.com/problems/max-chunks-to-make-sorted/solutions/6163196/kotlin-rust-by-samoylenkodmitry-kj71/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19122024-769-max-chunks-to-make-sorted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/eVGez7NRzA4)
[deep-dive](https://notebooklm.google.com/notebook/903899db-702f-4583-b779-b7af9d069781/audio)
![1.webp](/assets/leetcode_daily_images/1fdc8ae6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/837

#### Problem TLDR

Maximum independent chunks to merge-sort array #medium

#### Intuition

Let's observe when we can split the array:

```j

    // 0 1 2 3 4 5
    // 1 3 4 0 2 5
    // [       ][ ]
    // 1 3 4 0 5 2
    // [         ]
    //

```
Some observations:
* all numbers before split border should be less than the current index
* we should move the border up to the maximum of the seen values

#### Approach

* let's use `count`
* the problem size of `10` items hint for some brute-force DFS where we try every possible split and do the sort, however it is not the simplest way of solving

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxChunksToSorted(arr: IntArray): Int {
        var max = 0
        return (0..<arr.size).count {
            max = max(max, arr[it])
            it == max
        }
    }

```
```rust

    pub fn max_chunks_to_sorted(arr: Vec<i32>) -> i32 {
        let mut m = 0;
        (0..arr.len()).filter(|&i| {
            m = m.max(arr[i]); i as i32 == m
        }).count() as _
    }

```
```c++

    int maxChunksToSorted(vector<int>& arr) {
        int r = 0;
        for (int i = 0, m = 0; i < arr.size(); ++i) {
            m = max(m, arr[i]); if (i == m) r++;
        }
        return r;
    }

```

