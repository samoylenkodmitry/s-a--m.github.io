---
layout: leetcode-entry
title: "1146. Snapshot Array"
permalink: "/leetcode/problem/2023-06-11-1146-snapshot-array/"
leetcode_ui: true
entry_slug: "2023-06-11-1146-snapshot-array"
---

[1146. Snapshot Array](https://leetcode.com/problems/snapshot-array/description/) medium
[blog post](https://leetcode.com/problems/snapshot-array/solutions/3623764/kotlin-binary-search/)
[substack](https://dmitriisamoilenko.substack.com/p/11062023-1146-snapshot-array?sd=pf)
![image.png](/assets/leetcode_daily_images/bb748f77.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/242
#### Problem TLDR
Implement an array where all elements can be saved into a `snapshot's.
#### Intuition
Consider example:

```

// 0 1 2 3 4 5 6 <-- snapshot id
// 1 . . 2 . . 3 <-- value

```

When `get()(2)` called, `1` must be returned. So, we need to keep all the previous values. We can put them into a list combining with the current `snapshot id`: `(1,0), (2, 3), (3, 6)`. Then we can do a Binary Search and find the `highest_id >= id`.

#### Approach
For more robust Binary Search:
* use inclusive `lo`, `hi`
* check last condition `lo == hi`
* always write the result `ind = mid`

##### Complexity
- Time complexity:
$$O(log(n))$$ for `get`
- Space complexity:
$$O(n)$$

#### Code

```kotlin

class SnapshotArray(length: Int) {
    // 0 1 2 3 4 5 6
    // 1 . . 2 . . 3
    val arr = Array<MutableList<Pair<Int, Int>>>(length) { mutableListOf() }
    var currId = 0

    fun set(index: Int, v: Int) {
        val idVs = arr[index]
        if (idVs.isEmpty() || idVs.last().first != currId) idVs += currId to v
        else idVs[idVs.lastIndex] = currId to v
    }

    fun snap(): Int = currId.also { currId++ }

    fun get(index: Int, id: Int): Int {
        var lo = 0
        var hi = arr[index].lastIndex
        var ind = -1
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (arr[index][mid].first <= id) {
                ind = mid
                lo = mid + 1
            } else hi = mid - 1
        }
        return if (ind == -1) 0 else arr[index][ind].second
    }

}

```

