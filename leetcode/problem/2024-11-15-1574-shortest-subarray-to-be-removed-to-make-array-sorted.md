---
layout: leetcode-entry
title: "1574. Shortest Subarray to be Removed to Make Array Sorted"
permalink: "/leetcode/problem/2024-11-15-1574-shortest-subarray-to-be-removed-to-make-array-sorted/"
leetcode_ui: true
entry_slug: "2024-11-15-1574-shortest-subarray-to-be-removed-to-make-array-sorted"
---

[1574. Shortest Subarray to be Removed to Make Array Sorted](https://leetcode.com/problems/shortest-subarray-to-be-removed-to-make-array-sorted/description/) medium
[blog post](https://leetcode.com/problems/shortest-subarray-to-be-removed-to-make-array-sorted/solutions/6047263/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15112024-1574-shortest-subarray-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ShoQTGvRqiY)
[deep-dive](https://notebooklm.google.com/notebook/aea72038-d81e-4e98-ba84-6176811f715d/audio)
![1.webp](/assets/leetcode_daily_images/4c237922.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/801

#### Problem TLDR

Min subarray remove to make array sorted #medium #two_pointers

#### Intuition

(Failed)

There are only 3 possibilities:
* remove from the start
* remove from the end
* remove from the center

How to optimally remove from the center?

(At this point I've used all the hints and gave up)

For example: `1 2 3 4 1 1 3 2 3 4 5 6`
Take prefix until it is sorted `1 2 3 4`.
Take suffix until it is sorted `2 3 4 5 6`.
Now we have to optimally overlap it:

```j

1 2 3 4
    2 3 4 5 6

```

However, some overlaps are not obvious:

```j

1 2 3 3 3 3 4
    2 3 4 5 6 <-- not optimal
          3 4 5 6 <-- skip 2, optimal

```

So, we have to search though all possible overlaps and peek the best result.

(What was hard for me is to arrive to how exactly search all possible overlaps, my attempt to decrease left and increase right pointers was wrong)

The optimal way to do this is to scan both prefix and suffix from the start, always increasing the smallest one.

#### Approach

* micro-optimization: we can find prefix in the same loop as the main search

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun findLengthOfShortestSubarray(arr: IntArray): Int {
        var h = arr.lastIndex; var l = 0
        while (h > 0 && arr[h - 1] <= arr[h]) h--
        var res = h
        while (l < h && h <= arr.size && (l < 1 || arr[l] >= arr[l - 1]))
            if (h == arr.size || arr[l] <= arr[h])
            res = min(res, h - l++ - 1) else h++
        return res
    }

```
```rust

    pub fn find_length_of_shortest_subarray(arr: Vec<i32>) -> i32 {
        let n = arr.len(); let (mut l, mut h) = (0, n - 1);
        while h > 0 && arr[h - 1] <= arr[h] { h -= 1 }
        let mut res = h;
        while l < h && h <= n && (l < 1 || arr[l] >= arr[l - 1]) {
            if h == n || arr[l] <= arr[h] {
                res = res.min(h - l - 1); l += 1
            } else { h += 1 }
        }; res as i32
    }

```
```c++

    int findLengthOfShortestSubarray(vector<int>& arr) {
        int n = arr.size(), h = n - 1, res, l = 0;
        while (h > 0 && arr[h - 1] <= arr[h]) h--;
        for (res = h; l < h && h <= n && (l < 1 || arr[l] >= arr[l - 1]);)
            h == n || arr[l] <= arr[h] ? res = min(res, h - l++ - 1) : h++;
        return res;
    }

```

