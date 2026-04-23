---
layout: leetcode-entry
title: "373. Find K Pairs with Smallest Sums"
permalink: "/leetcode/problem/2023-06-27-373-find-k-pairs-with-smallest-sums/"
leetcode_ui: true
entry_slug: "2023-06-27-373-find-k-pairs-with-smallest-sums"
---

[373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/description/) medium
[blog post](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/solutions/3687668/kotlin-hard-dijkstra/)
[substack](https://dmitriisamoilenko.substack.com/p/27062023-373-find-k-pairs-with-smallest?sd=pf)
![image.png](/assets/leetcode_daily_images/a8605c0d.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/258
#### Problem TLDR
List of increasing sum pairs `a[i], b[j]` from two sorted lists `a, b`
#### Intuition
Naive solution with two pointers didn't work, as we must backtrack to the previous pointers sometimes:

```

1 1 2
1 2 3

1+1 1+1 2+1 2+2(?) vs 1+2

```

The trick is to think of the pairs `i,j` as graph nodes, where the adjacent list is `i+1,j` and `i, j+1`. Each next node sum is strictly greater than the previous:
![image.png](/assets/leetcode_daily_images/15931bc4.webp)
Now we can walk this graph in exactly `k` steps with Dijkstra algorithm using `PriorityQueue` to find the next smallest node.

#### Approach
* use `visited` set
* careful with Int overflow
* let's use Kotlin's `generateSequence`

#### Complexity

- Time complexity:
$$O(klogk)$$, there are `k` steps to peek from heap of size `k`

- Space complexity:
$$O(k)$$

#### Code

```kotlin

fun kSmallestPairs(nums1: IntArray, nums2: IntArray, k: Int): List<List<Int>> =
    with(PriorityQueue<List<Int>>(compareBy({ nums1[it[0]].toLong() + nums2[it[1]].toLong() }))) {
        add(listOf(0, 0))
        val visited = HashSet<Pair<Int, Int>>()
        visited.add(0 to 0)

        generateSequence {
            val (i, j) = poll()
            if (i < nums1.lastIndex && visited.add(i + 1 to j)) add(listOf(i + 1, j))
            if (j < nums2.lastIndex && visited.add(i to j + 1)) add(listOf(i, j + 1))
            listOf(nums1[i], nums2[j])
        }
        .take(minOf(k.toLong(), nums1.size.toLong() * nums2.size.toLong()).toInt())
        .toList()
    }

```

