---
layout: post
title: Daily leetcode challenge
---

# Daily leetcode challenge
You can join me and discuss in the Telegram channel [https://t.me/leetcode_daily_unstoppable](https://t.me/leetcode_daily_unstoppable)

*If you use this text to train artificial intelligence, you must share the final product with me to use it for free*
#### You can donate me in crypto:
* xmr 84rsnuoKbHKVGVaT1Z22YQahSuBJKDYmGjQuHYkv637VApfHPR4oj2eAtYCERFQRvnQWRV8UWBDHTUhmYXf8qyo8F33neiH
* btc bc1qj4ngpjexw7hmzycyj3nujjx8xw435mz3yflhhq
* doge DEb3wN29UCYvfsiv1EJYHpGk6QwY4HMbH7

# 21.07.2023
[673. Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/) medium
[blog post](https://leetcode.com/problems/number-of-longest-increasing-subsequence/solutions/3795250/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/21072023-673-number-of-longest-increasing?sd=pf)
![image.png](https://assets.leetcode.com/users/images/0b5786e0-849b-4852-b131-13bd9813fd94_1689915416.2290564.png)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/282

#### Proble TLDR

Count of LIS in an array

#### Intuition

To find Longest Increasing Subsequence, there is a known algorithm with $$O(nlog(n))$$ time complexity. However, it can help with this case:

```bash

3 5 4 7

```

when we must track both `3 4 7` and `3 5 7` sequences. Given that, we can try to do full search with DFS, taking or skipping a number. To cache some results, we must make `dfs` depend on only the input arguments. Let's define it to return both `max length of LIS` and `count of them` in one result, and arguments are the starting position in an array and `previous number` that we must start sequence from.

#### Approach 

* use an array cache, as `Map` gives TLE

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```

    class R(val maxLen: Int, val cnt: Int)
    fun findNumberOfLIS(nums: IntArray): Int {
      val cache = Array(nums.size + 1) { Array<R>(nums.size + 2) { R(0, 0) } }
      fun dfs(pos: Int, prevPos: Int): R = if (pos == nums.size) R(0, 1) else 
        cache[pos][prevPos].takeIf { it.cnt != 0 }?: {
          val prev = if (prevPos == nums.size) Int.MIN_VALUE else nums[prevPos]
          var cnt = 0
          while (pos + cnt < nums.size && nums[pos + cnt] == nums[pos]) cnt++
          val skip = dfs(pos + cnt, prevPos)
          if (nums[pos] <= prev) skip else {
            val start = dfs(pos + cnt, pos).let { R(1 + it.maxLen, cnt * it.cnt ) }
            if (skip.maxLen == start.maxLen) R(skip.maxLen, start.cnt + skip.cnt)
            else if (skip.maxLen > start.maxLen) skip else start
          }
        }().also { cache[pos][prevPos] = it }
      return dfs(0, nums.size).cnt
    }

```

#### Magical rundown

```
🏰🔮🌌 The Astral Enigma of Eternity
In the boundless tapestry of time, an enigmatic labyrinth 🗝️ whispers
tales of forgotten epochs. Your fateful quest? To decipher the longest
increasing subsequences hidden within the celestial array 🧩 [3, 5, 4, 7].

🌄 The Aurora Gateway: dfs(0, nums.size)
    /                          \
🌳 The Verdant Passage (dfs(1,0)) / 🌑 The Nebulous Veil (dfs(1,nums.size))

Your odyssey commences at twilight's brink: will you tread the lush
🌳 Verdant Passage or dare to penetrate the enigmatic 🌑 Nebulous Veil?

🌄 The Aurora Gateway: dfs(0, nums.size)
   /   
🍃 The Glade of Whispers (Pos 1: num[1]=3, dfs(1,0))  
   /  
🌊 The Cascade of Echoes (Pos 2: num[2]=5, dfs(2,1))  
   / 
⛰️ The Bastion of Silence (Pos 3: num[3]=4, dfs(3,2)) 🚫🔒

The labyrinth’s heart pulsates with cryptic riddles. The ⛰️ Bastion of Silence
remains locked, overshadowed by the formidable 🌊 Cascade of Echoes.

🌄 The Aurora Gateway: dfs(0, nums.size)
   /   
🍃 The Glade of Whispers (Pos 1: num[1]=3, dfs(1,0))  
   \   
🌑 The Phantom of Riddles (Pos 2: num[2]=5, dfs(2,0)) 

Retracing your footsteps, echoes of untaken paths whisper secrets. Could
the ⛰️ Bastion of Silence hide beneath the enigma of the 🌑 Phantom of Riddles?

🌄 The Aurora Gateway: dfs(0, nums.size)
   /   
🍃 The Glade of Whispers (Pos 1: num[1]=3, dfs(1,0))  
   \   
💨 The Mist of Mystery (Pos 3: num[3]=4, dfs(3,0))
   \
🌩️ The Tempest of Triumph (Pos 4: num[4]=7, dfs(4,3)) 🏁🎉

At last, the tempest yields! Each twist and turn, each riddle spun and
secret learned, illuminates a longest increasing subsequence in the cosmic array.

Your enchanted grimoire 📜✨ (cache) now vibrates with the wisdom of ages:

prevPos\pos  0     1      2      3     4
       0     (0,0) (2,1) (2,1)  (3,2) (0,0)
       1     (0,0) (0,0) (2,1)  (3,2) (0,0)
       2     (0,0) (0,0) (0,0)  (2,1) (0,0)
       3     (0,0) (0,0) (0,0)  (0,0) (0,0)
       4     (0,0) (0,0) (0,0)  (0,0) (0,0)

Beneath the shimmering cosmic symphony, you cast the final incantation
🧙‍♂️ dfs(0, nums.size).cnt. The grimoire blazes with ethereal light, revealing
the total count of longest increasing subsequences.

You emerge from the labyrinth transformed: no longer merely an adventurer,
but the 🌟 Cosmic Guardian of Timeless Wisdom. 🗝️✨🌠

```


# 20.07.2023
[735. Asteroid Collision](https://leetcode.com/problems/asteroid-collision/description/) medium
[blog post](https://leetcode.com/problems/asteroid-collision/solutions/3790443/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/20072023-735-asteroid-collision?sd=pf)
![image.png](https://assets.leetcode.com/users/images/2b671792-cf2a-4c2d-907c-0b79dd627b9e_1689826275.7227218.png)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/281

#### Problem TLDR

Result after asteroids collide left-right exploding by size: `15 5 -15 -5 5 -> -15 -5 5`

#### Intuition

Let's add positive asteroids to the `Stack`. When negative met, it can fly over all smaller positive added, and can explode if larger met.

#### Approach

Kotlin's API helping reduce some LOC

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$


#### Code

```kotlin

    fun asteroidCollision(asteroids: IntArray): IntArray = with(Stack<Int>()) {
        asteroids.forEach { sz ->
          if (!generateSequence { if (sz > 0 || isEmpty() || peek() < 0) null else peek() }
            .any {
              if (it <= -sz) pop()
              it >= -sz
            }) add(sz)
        }
        toIntArray()
    }

```

# 19.07.2023
[435. Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/description/) medium
[blog post](https://leetcode.com/problems/non-overlapping-intervals/solutions/3785669/kotlin-line-sweep/)
[substack](https://dmitriisamoilenko.substack.com/p/19072023-435-non-overlapping-intervals?sd=pf)
![image.png](https://assets.leetcode.com/users/images/3f9e85af-5956-4212-a56e-2f201030a2aa_1689738344.4310584.png)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/280

#### Problem TLDR

Minimum intervals to erase overlap

#### Intuition

First idea, is to sort the array by `from`. Next, we can greedily take intervals and remove overlapping ones. But, to remove the `minimum` number, we can start with removing the most `long` intervals.

#### Approach

* walk the sweep line, counting how many intervals are non overlapping
* only move the `right border` when there is a new non overlapping interval
* minimize the `border` when it shrinks

#### Complexity


- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun eraseOverlapIntervals(intervals: Array<IntArray>): Int {
        intervals.sortWith(compareBy({ it[0] }))
        var border = Int.MIN_VALUE
        return intervals.count { (from, to) -> 
          (border > from).also {
            if (border <= from || border > to) border = to
          }
        }
    }

```

# 18.07.2023
[146. LRU Cache](https://leetcode.com/problems/lru-cache/description/) medium
[blog post](https://leetcode.com/problems/lru-cache/solutions/3781121/kotlin-linked-list/)
[substack](https://dmitriisamoilenko.substack.com/p/18072023-146-lru-cache?sd=pf)
![image.png](https://assets.leetcode.com/users/images/23d9fff8-2793-4ee5-afa9-6f3788537668_1689652989.7052531.png)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/279

#### Intuition

We can use Doubly-Linked List representing access time in its order.

#### Approach

* use `firstNode` and `lastNode`

#### Complexity

- Time complexity:
$$O(1)$$, for each call `get` or `put`

- Space complexity:
$$O(1)$$, for each element

#### Code

```kotlin

class LRUCache(val capacity: Int) {
    class Node(val key: Int, var left: Node? = null, var right: Node? = null)
    var size = 0
    val map = mutableMapOf<Int, Int>()
    val firstNode = Node(-1)
    var lastNode = firstNode
    val keyToNode = mutableMapOf<Int, Node>()

    fun disconnect(node: Node) {
      val leftNode = node.left
      val rightNode = node.right
      node.left = null
      node.right = null
      leftNode?.right = rightNode
      rightNode?.left = leftNode
      if (node === lastNode) lastNode = leftNode!!
    }

    fun updateNode(key: Int) {
      val node = keyToNode[key]!!
      if (node === lastNode) return
      disconnect(node)
      lastNode.right = node
      node.left = lastNode
      lastNode = node
    }

    fun get(key: Int): Int = map[key]?.also { updateNode(key) } ?: -1

    fun put(key: Int, value: Int) {
      if (!map.contains(key)) {
        if (size == capacity) {
          firstNode.right?.let {
            map.remove(it.key)
            keyToNode.remove(it.key)
            disconnect(it)
          }
        } else size++
        keyToNode[key] = Node(key)
      }
      updateNode(key)
      map[key] = value
    }

}

```

# 17.07.2023
[445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/description/) medium
[blog post](https://leetcode.com/problems/add-two-numbers-ii/solutions/3776193/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/17072023-445-add-two-numbers-ii)
![image.png](https://assets.leetcode.com/users/images/868d7f62-48ba-4adf-a8f0-fbba3dbcc01a_1689566938.506953.png)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/278

#### Problem TLDR

Linked List of sum of two Linked Lists numbers, `9->9 + 1 = 1->0->0`

#### Intuition

The hint is in the description: reverse lists, then just do arithmetic. Another way is to use stack.

#### Approach

* don't forget to undo the reverse

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun addTwoNumbers(l1: ListNode?, l2: ListNode?, n: Int = 0): ListNode? {
      fun ListNode?.reverse(): ListNode? {
        var curr = this
        var prev: ListNode? = null
        while (curr != null) {
          val next = curr.next
          curr.next = prev
          prev = curr
          curr = next
        }
        return prev
      }
      var l1r = l1.reverse()
      var l2r = l2.reverse()
      var o = 0
      var prev: ListNode? = null
      while (l1r != null || l2r != null) {
        val v = o + (l1r?.`val` ?: 0) + (l2r?.`val` ?: 0)
        prev = ListNode(v % 10).apply { next = prev }
        o = v / 10
        l1r = l1r?.next
        l2r = l2r?.next
      }
      if (o > 0) prev = ListNode(o).apply { next = prev }
      l1r.reverse()
      l2r.reverse()
      return prev
    }

```

# 16.07.2023
[1125. Smallest Sufficient Team](https://leetcode.com/problems/smallest-sufficient-team/description/) hard
[blog post](https://leetcode.com/problems/smallest-sufficient-team/solutions/3771197/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/16072023-1125-smallest-sufficient?sd=pf)

![image.png](https://assets.leetcode.com/users/images/6ff98105-4fdb-4d51-a086-31ddf36f4ebc_1689492977.1362433.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/277

#### Problem TLDR
Smallest `team` from `people with skills`, having all `required skills`

#### Intuition
The skills set size is less than `32`, so we can compute a `bitmask` for each of `people` and for the `required` skills.
Next, our task is to choose a set from `people` that result skills mask will be equal to the `required`.
We can do a full search, each time `skipping` or `adding` one mask from the `people`.  
Observing the problem, we can see, that result is only depending on the current `mask` and all the `remaining` people. So, we can cache it.

#### Approach
* we can use a `HashMap` to store `skill to index`, but given a small set of skills, just do `indexOf` in O(60 * 16)
* add to the team in `post order`, as `dfs` must return only the result depending on the input arguments

#### Complexity

- Time complexity:
$$O(p2^s)$$, as full mask bits are 2^s, s - skills, p - people

- Space complexity:
$$O(p2^s)$$

#### Code

```kotlin

    fun smallestSufficientTeam(skills: Array<String>, people: List<List<String>>): IntArray {
        val peoplesMask = people.map {  it.fold(0) { r, t -> r or (1 shl skills.indexOf(t)) } }
        val cache = mutableMapOf<Pair<Int, Int>, List<Int>>()
        fun dfs(curr: Int, mask: Int): List<Int> =
          if (mask == (1 shl skills.size) - 1) listOf()
          else if (curr == people.size) people.indices.toList()
          else cache.getOrPut(curr to mask) {
            val skip = dfs(curr + 1, mask)
            val take = dfs(curr + 1, mask or peoplesMask[curr]) + curr
            if (skip.size < take.size) skip else take
          }
        return dfs(0, 0).toIntArray()
    }

```

# 15.07.2023
[1751. Maximum Number of Events That Can Be Attended II](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/description/) hard
[blog post](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/solutions/3766779/kotln-dp/)
[substack](https://dmitriisamoilenko.substack.com/p/15072023-1751-maximum-number-of-events?sd=pf)
![image.png](https://assets.leetcode.com/users/images/5d01488f-193e-46e7-8910-fcdbaec93d00_1689394389.4020226.png)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/276

#### Problem TLDR

Max sum of at most `k` `values` from non-intersecting array of `(from, to, value)` items

#### Intuition

Let's observe example:

```bash
        // 0123456789011
        // [     4 ]
        // [1][2][3][2]
        //      [4][2]

```

If `k=1` we choose `[4]`
if `k=2` we choose `[4][2]`
if `k=3` we choose `[2][3][2]`

###### What will not work:

* sweep line algorithm, as it is greedy, but there is an only `k` items we must choose and we must do backtracking
* adding to Priority Queue and popping the lowest values: same problem, we must backtrack

###### What will work:

* asking for a hint: this is what I used
* full search: at every `index` we can `pick` or `skip` the element
* sorting: it will help to reduce irrelevant combinations by doing a Binary Search for the next non-intersecting element

We can observe, that at any given position the result only depends on the suffix array. That means we can safely cache the result by the current position.

#### Approach

For more robust Binary Search code:
* use inclusive `lo`, `hi`
* check the last condition `lo == hi`
* always write the result `next = mid`
* always move the borders `lo = mid + 1`, `hi = mid - 1`

#### Complexity

- Time complexity:
$$O(nklog(n))$$

- Space complexity:
$$O(nk)$$

#### Code

```kotlin

    fun maxValue(events: Array<IntArray>, k: Int): Int {
        // 0123456789011
        // [     4 ]
        // [1][2][3][2]
        //      [4][2]
        val inds = events.indices.sortedWith(compareBy({ events[it][0] }))
        // my ideas: 
        // sort - good
        // sweep line ? - wrong
        // priority queue ? - wrong
        // binary search ? 1..k - wrong
        // used hints:
        // hint: curr + next vs drop  dp?
        // hint: binary search next
        val cache = mutableMapOf<Pair<Int, Int>, Int>()
        fun dfs(curr: Int, canTake: Int): Int {
          return if (curr ==  inds.size || canTake == 0) 0
          else cache.getOrPut(curr to canTake) {
            val (_, to, value) = events[inds[curr]]
            var next = inds.size
            var lo = curr + 1
            var hi = inds.lastIndex
            while (lo <= hi) {
              val mid = lo + (hi - lo) / 2
              val (nextFrom, _, _) = events[inds[mid]]
              if (nextFrom > to) {
                next = mid
                hi = mid - 1
              } else lo = mid + 1
            }
            maxOf(value + dfs(next, canTake - 1), dfs(curr + 1, canTake))
          }
        }
        return dfs(0, k)
    }

```

# 14.07.2023
[1218. Longest Arithmetic Subsequence of Given Difference](https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/description/) medium
[blog post](https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/solutions/3761793/kotlin-map/)
[substack](https://dmitriisamoilenko.substack.com/p/14072023-1218-longest-arithmetic?sd=pf)
![image.png](https://assets.leetcode.com/users/images/66057f03-69d5-4709-9142-79fc3d54720e_1689304858.803253.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/275
#### Problem TLDR
Longest arithmetic `difference` subsequence
#### Intuition
Store the `next` value and the `length` for it.

#### Approach
We can use a `HashMap`
#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun longestSubsequence(arr: IntArray, difference: Int): Int = 
with(mutableMapOf<Int, Int>()) {
    arr.asSequence().map { x ->
        (1 + (this[x] ?: 0)).also { this[x + difference] = it } 
    }.max()!!
}

```

# 13.07.2023
[207. Course Schedule](https://leetcode.com/problems/course-schedule/description/) medium
[blog post](https://leetcode.com/problems/course-schedule/solutions/3757355/kotlin-toposort-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/13072023-207-course-schedule?sd=pf)
![image.png](https://assets.leetcode.com/users/images/b9681eb4-001e-4cf5-a086-135b40d9f474_1689219966.714815.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/274
#### Problem TLDR
If `none` edges in a cycle
#### Intuition
To detect cycle, we can use DFS and two sets `cycle` and `safe`. Or use Topological Sort and check that all elements are visited.

#### Approach
Let's use Topological Sort with Breadth-First Search.
* build `indegree` - number of input nodes for each node
* add to BFS only nodes with `indegree[node] == 0`
* decrease `indegree` as it visited

#### Complexity
- Time complexity:
$$O(VE)$$

- Space complexity:
$$O(E + V)$$

#### Code

```kotlin

fun canFinish(numCourses: Int, prerequisites: Array<IntArray>): Boolean {
    val fromTo = mutableMapOf<Int, MutableSet<Int>>()
        val indegree = IntArray(numCourses)
        prerequisites.forEach { (to, from) ->
            fromTo.getOrPut(from) { mutableSetOf() } += to
            indegree[to]++
        }
        return with(ArrayDeque<Int>()) {
            addAll((0 until numCourses).filter { indegree[it] == 0 })
            generateSequence { if (isEmpty()) null else poll() }.map {
                fromTo[it]?.forEach {
                    if (--indegree[it] == 0) add(it)
                }
            }.count() == numCourses
        }
    }

```

# 12.07.2023
[802. Find Eventual Safe States](https://leetcode.com/problems/find-eventual-safe-states/description/) medium
[blog post](https://leetcode.com/problems/find-eventual-safe-states/solutions/3752760/kotlin-dfs/)
[substack](https://dmitriisamoilenko.substack.com/p/13072023-802-find-eventual-safe-states?sd=pf)
![image.png](https://assets.leetcode.com/users/images/89561214-ce93-4759-8181-accd708139ea_1689134618.718084.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/273
#### Problem TLDR
List of nodes not in cycles
#### Intuition
Simple Depth-First Search will give optimal $$O(n)$$ solution.
When handling the `visited` set, we must separate those in `cycle` and `safe`.
#### Approach
* we can remove from `cycle` set and add to `safe` set in a post-order traversal

#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun eventualSafeNodes(graph: Array<IntArray>): List<Int> {
    val cycle = mutableSetOf<Int>()
        val safe = mutableSetOf<Int>()
            fun cycle(curr: Int): Boolean {
                return if (safe.contains(curr)) false else !cycle.add(curr)
                || graph[curr].any { cycle(it) }
                .also {
                    if (!it) {
                        cycle.remove(curr)
                        safe.add(curr)
                    }
                }
            }
            return graph.indices.filter { !cycle(it) }
        }

```

# 12.07.2023
[863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/solutions/3748155/kotlin-dfs-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/12072023-863-all-nodes-distance-k?sd=pf)
![image.png](https://assets.leetcode.com/users/images/d76b7c73-241b-4a19-9c94-be38b96ba456_1689048976.2239547.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/272
#### Problem TLDR
List of `k` distanced from `target` nodes in a Binary Tree
#### Intuition
There is a one-pass DFS solution, but it feels like too much of a corner cases and result handholding.
A more robust way is to traverse with DFS and connect children nodes to parent, then send a wave from target at `k` steps.

#### Approach
Let's build an undirected graph and do BFS.
* don't forget a visited `HashSet`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun distanceK(root: TreeNode?, target: TreeNode?, k: Int): List<Int> {
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        fun dfs(node: TreeNode?, parent: TreeNode?) {
            node?.run {
                parent?.let {
                    fromTo.getOrPut(`val`) { mutableListOf() } += it.`val`
                    fromTo.getOrPut(it.`val`) { mutableListOf() } += `val`
                }
                dfs(left, this)
                dfs(right, this)
            }
        }
        dfs(root, null)
        return LinkedList<Int>().apply {
            val visited = HashSet<Int>()
                target?.run {
                    add(`val`)
                    visited.add(`val`)
                }
                repeat(k) {
                    repeat(size) {
                        fromTo.remove(poll())?.forEach { if (visited.add(it)) add(it) }
                    }
                }
            }
        }

```

# 11.07.2023
[111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/description/) easy
[blog post](https://leetcode.com/problems/minimum-depth-of-binary-tree/solutions/3743369/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/11072023-111-minimum-depth-of-binary?sd=pf)
![image.png](https://assets.leetcode.com/users/images/9496f18c-1cdd-4224-9ed9-2ae8d5099c44_1688960338.7698486.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/271
#### Problem TLDR
Count nodes in the shortest path from root to leaf
#### Intuition
* remember to count `nodes`, not `edges`
* `leaf` is a node without children
* use BFS or DFS

#### Approach
Let's use BFS

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun minDepth(root: TreeNode?): Int = with(ArrayDeque<TreeNode>()) {
    root?.let { add(it) }
    generateSequence(1) { (it + 1).takeIf { isNotEmpty() } }
    .firstOrNull {
        (1..size).any {
            with(poll()) {
                left?.let { add(it) }
                right?.let { add(it) }
                left == null && right == null
            }
        }
    } ?: 0
}

```

# 10.07.2023
[2272. Substring With Largest Variance](https://leetcode.com/problems/substring-with-largest-variance/description/) hard
[blog post](https://leetcode.com/problems/substring-with-largest-variance/solutions/3739542/kotlin-try-all-pairs/)
[substack](https://dmitriisamoilenko.substack.com/p/10072023-2272-substring-with-largest?sd=pf)
![image.png](https://assets.leetcode.com/users/images/c0c1f372-45e0-4a71-a86b-cc062582f0e9_1688881095.5316043.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/270
#### Problem TLDR
Max diff between count `s[i]` and count `s[j]` in all substrings of `s`
#### Intuition
The first idea is to simplify the task by considering only two chars, iterating over all alphabet combinations.
Second idea is how to solve this problem for binary string in $$O(n)$$: `abaabbb` → `abbb`.
We split this problem: find the largest subarray for `a` with the smallest count of `b`, and reverse the problem – largest `b` with smallest `a`.
For this issue, there is a Kadane's algorithm for maximizing `sum`: take values greedily and reset count when `sum < 0`.
Important customization is to always consider `countB` at least `1` as it must be present in a subarray.

#### Approach
* we can use `Set` of only the chars in `s`
* iterate in `ab` and `ba` pairs
* Kotlin API helps save some LOC

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, or O(1) if `asSequence` used

#### Code

```kotlin

fun largestVariance(s: String): Int = s.toSet()
.let { ss -> ss.map { a -> ss.filter { it != a }.map { a to it } }.flatten() }
.map { (a, b) ->
    var countA = 0
    var countB = 0
    s.filter { it == a || it == b }
    .map { c ->
        if (c == a) countA++ else countB++
        if (countA < countB) {
            countA = 0
            countB = 0
        }
        countA - maxOf(1, countB)
    }.max() ?: 0
}.max() ?: 0

```

# 9.07.2023
[2551. Put Marbles in Bags](https://leetcode.com/problems/put-marbles-in-bags/description/) hard
[blog post](https://leetcode.com/problems/put-marbles-in-bags/solutions/3734482/kotlin-priorityqueue/)
[substack](https://dmitriisamoilenko.substack.com/p/9072023-2551-put-marbles-in-bags?sd=pf)
![image.png](https://assets.leetcode.com/users/images/0266de8c-9c1b-4ebf-87ac-b370530136f4_1688788395.230992.png)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/269
#### Problem TLDR
`abs(max - min)`, where `max` and `min` are the sum of `k` interval borders
#### Intuition
Let's observe some examples:

```

// 1 3 2 3 5 4 5 7 6
// *   * *
// 1+3 2+2 3+6 = 4+4+9 = 17
// * * *
// 1+1 3+3 2+6 = 2+6+8 = 16
// *             * * = 1+5 7+7 6+6
// 1 9 1 9 1 9 1 9 1    k = 3
// *   *           *    s = 1+9+1+9+1+1
// * *   *              s = 1+1+9+1+9+1
// 1 1 9 9 1 1 9 9 1    k = 3
// * *       *          s = 1+1+1+1+1+1
// *     *       *      s = 1+9+9+9+9+1
// 1 1 1 9 1 9 9 9 1    k = 3
// * * *                s = 1+1+1+1+1+1
// *         . * *      s = 1+9+9+9+9+1
// 1 4 2 5 2            k = 3
// . * . *              1+1+4+2+5+2
//   . * *              1+4+2+2+5+2
// . *   . *            1+1+4+5+2+2

```

One thing to note, we must choose `k-1` border pairs `i-1, i` with `min` or `max` sum.

#### Approach
Let's use PriorityQueue.

#### Complexity

- Time complexity:
$$O(nlog(k))$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin

fun putMarbles(weights: IntArray, k: Int): Long {

    val pqMax = PriorityQueue<Int>(compareBy( { weights[it].toLong() + weights[it - 1].toLong() } ))
        val pqMin = PriorityQueue<Int>(compareByDescending( { weights[it].toLong() + weights[it - 1].toLong() } ))
            for (i in 1..weights.lastIndex) {
                pqMax.add(i)
                if (pqMax.size > k - 1) pqMax.poll()
                pqMin.add(i)
                if (pqMin.size > k - 1) pqMin.poll()
            }
            return Math.abs(pqMax.map { weights[it].toLong() + weights[it - 1].toLong() }.sum()!! -
            pqMin.map { weights[it].toLong() + weights[it - 1].toLong() }.sum()!!)
        }

```

# 7.07.2023
[2024. Maximize the Confusion of an Exam](https://leetcode.com/problems/maximize-the-confusion-of-an-exam/description/) medium
[blog post](https://leetcode.com/problems/maximize-the-confusion-of-an-exam/solutions/3730076/kotlin-sliding-window/)
[substack](https://dmitriisamoilenko.substack.com/p/7072023-2024-maximize-the-confusion?sd=pf)
![image.png](https://assets.leetcode.com/users/images/dddfd1fa-9dd9-4e9f-8c7a-1f142a5f8fb8_1688702846.7571077.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/268
#### Problem TLDR
Max same letter subarray replacing `k` letters
#### Intuition
An important example is `ftftftft k=3`: we must fill all the intervals. It also tells, after each filling up we must decrease `k`. Let's count `T` and `F`.
Sliding window is valid when `tt <= k || ff <= k`.
#### Approach
We can save some lines using Kotlin collections API

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, or $$O(1)$$ using `asSequence`

#### Code

```kotlin

fun maxConsecutiveAnswers(answerKey: String, k: Int): Int {
    var tt = 0
    var ff = 0
    var lo = 0
    return answerKey.mapIndexed { i, c ->
        if (c == 'T') tt++ else ff++
        while (tt > k && ff > k && lo < i)
        if (answerKey[lo++] == 'T') tt-- else ff--
        i - lo + 1
    }.max() ?: 0
}

```

# 6.07.2023
[209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/) medium
[blog post](https://leetcode.com/problems/minimum-size-subarray-sum/solutions/3724899/kotlin-two-pointers/)
[substack](https://dmitriisamoilenko.substack.com/p/6072023-209-minimum-size-subarray?sd=pf)
![image.png](https://assets.leetcode.com/users/images/cdf31d10-015c-4bf8-8c53-71de5b2886b5_1688614271.2972152.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/267
#### Problem TLDR
Min length subarray with `sum >= target`
#### Intuition
Use two pointers: one adding to `sum` and another subtracting. As all numbers are positive, then `sum` will always be increasing with adding a number and deceasing when subtracting.

#### Approach
Let's use Kotlin `Sequence` API

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun minSubArrayLen(target: Int, nums: IntArray): Int {
    var lo = 0
    var sum = 0
    return nums.asSequence().mapIndexed { hi, n ->
        sum += n
        while (sum - nums[lo] >= target) sum -= nums[lo++]
        (hi - lo + 1).takeIf { sum >= target }
    }
    .filterNotNull()
    .min() ?: 0
}

```

# 5.07.2023
[1493. Longest Subarray of 1's After Deleting One Element](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/) medium
[blog post](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/solutions/3720190/kotlin-3-pointers/)
[substack](https://dmitriisamoilenko.substack.com/p/5072023-1493-longest-subarray-of?sd=pf)
![image.png](https://assets.leetcode.com/users/images/39baf54e-ae69-4ce8-b5a5-37781e40fd50_1688531738.628173.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/266
#### Problem TLDR
Largest `1..1` subarray after removing one item
#### Intuition
Let's maintain two pointers for a `start` and a `nextStart` positions, and a third pointer for the `right` border.

* move `start` to the `nextStart` when `right` == 0
* move `nextStart` to start of `1`'s

#### Approach
* corner case is when all array is `1`'s, as we must remove `1` then anyway

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ add `asSequence` for it to become $$O(1)$$

#### Code

```kotlin

fun longestSubarray(nums: IntArray): Int {
    var start = -1
    var nextStart = -1
    return if (nums.sum() == nums.size) nums.size - 1
    else nums.mapIndexed { i, n ->
        if (n == 0) {
            start = nextStart
            nextStart = -1
            0
        } else {
            if (nextStart == -1) nextStart = i
            if (start == -1) start = nextStart
            i - start + (if (start == nextStart) 1 else 0)
        }
    }.max() ?:0
}

```

# 4.07.2023
[137. Single Number II](https://leetcode.com/problems/single-number-ii/solutions/) medium
[blog post](https://leetcode.com/problems/single-number-ii/solutions/3715279/kotlin-o-32n/)
[substack](https://dmitriisamoilenko.substack.com/p/4072023-137-single-number-ii?sd=pf)
![image.png](https://assets.leetcode.com/users/images/6753bd27-4d46-494f-a5df-d1e902ced21d_1688442287.7725842.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/265
#### Proble TLDR
Single number in an array of tripples
#### Intuition
One simple approach it to count bits at each position.
Result will have a `1` when `count % 3 != 0`.

#### Approach
Let's use fold.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun singleNumber(nums: IntArray): Int =
//110
//110
//110
//001
//001
//001
//010
//010
//010
//100
//463
(0..31).fold(0) { res, bit ->
    res or ((nums.count { 0 != it and (1 shl bit) } % 3) shl bit)
}

```

# 3.07.2023
[859. Buddy Strings](https://leetcode.com/problems/buddy-strings/description/) easy
[blog post](https://leetcode.com/problems/buddy-strings/solutions/3710751/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/3072023-859-buddy-strings?sd=pf)
![image.png](https://assets.leetcode.com/users/images/45ae34ff-b248-46b6-8ab6-d9df90d58a8c_1688355711.8449478.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/264
#### Problem TLDR
Is it just one swap `s[i]<>s[j]` to string `s` == string `goal`
#### Intuition
Compare two strings for each position. There are must be only two not equal positions and they must be mirrored pairs.

#### Approach
Let's write it in Kotlin collections API style.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun buddyStrings(s: String, goal: String): Boolean = s.length == goal.length && (
s == goal && s.groupBy { it }.any { it.value.size > 1 } ||
s.zip(goal)
.filter { (a, b) -> a != b }
.windowed(2)
.map { (ab, cd) -> listOf(ab, cd.second to cd.first) }
.let { it.size == 1 && it[0][0] == it[0][1] }
)

```

# 2.07.2023
[1601. Maximum Number of Achievable Transfer Requests](https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests/description/) hard
[blog post](https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests/solutions/3706324/kotlin-bitmask/)
[substack](https://dmitriisamoilenko.substack.com/p/2072023-1601-maximum-number-of-achievable?sd=pf)
![image.png](https://assets.leetcode.com/users/images/9bd83a15-23b5-4715-a2f0-a77671903184_1688270064.1856298.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/263
#### Problem TLDR
Max edges to make all counts `in == out` edges in graph
#### Intuition
Let's observe some examples:
![image.png](https://assets.leetcode.com/users/images/23364750-5955-429c-bb95-c10e249de6fe_1688270255.5765126.png)

All requests are valid if count of incoming edges are equal to outcoming.
One possible solution is to just check each combination of edges.
#### Approach
Let's use bitmask to traverse all combinations, as total number `16` can fit in `Int`

#### Complexity

- Time complexity:
$$O(n2^r)$$

- Space complexity:
$$O(n2^r)$$

#### Code

```kotlin

fun maximumRequests(n: Int, requests: Array<IntArray>): Int =
    (0..((1 shl requests.size) - 1)).filter { mask ->
        val fromTo = IntArray(n)
        requests.indices.filter { ((1 shl it) and mask) != 0 }.forEach {
            val (from, to) = requests[it]
            fromTo[from] -= 1
            fromTo[to] += 1
        }
        fromTo.all { it == 0 }
    }.map { Integer.bitCount(it) }.max()!!

```

# 1.07.2023
[2305. Fair Distribution of Cookies](https://leetcode.com/problems/fair-distribution-of-cookies/description/) medium
[blog post](https://leetcode.com/problems/fair-distribution-of-cookies/solutions/3702635/kotln-backtrack/)
[substack](https://dmitriisamoilenko.substack.com/p/1072023-2305-fair-distribution-of?sd=pf)
![image.png](https://assets.leetcode.com/users/images/78843ab2-ca67-455a-9f8b-7e2550a2789f_1688186341.2668977.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/262

#### Problem TLDR
`Min` of the `max` distributing `n` cookies to `k` children
#### Intuition
Search all possible ways to give current cookie to one of the children. Backtrack sums and calculate the result.

#### Approach
Just DFS

#### Complexity

- Time complexity:
$$O(k^n)$$

- Space complexity:
$$O(2^n)$$

#### Code

```kotlin

fun distributeCookies(cookies: IntArray, k: Int): Int {
    fun dfs(pos: Int, children: IntArray): Int {
        if (pos == cookies.size) return if (children.contains(0)) -1 else children.max()!!
        var min = -1
        for (i in 0 until k) {
            children[i] += cookies[pos]
            val res = dfs(pos + 1, children)
            if (res != -1) min = if (min == -1) res else minOf(min, res)
            children[i] -= cookies[pos]
        }
        return min
    }
    return dfs(0, IntArray(k))
}

```

# 30.06.2023
[1970. Last Day Where You Can Still Cross](https://leetcode.com/problems/last-day-where-you-can-still-cross/description/) hard
[blog post](https://leetcode.com/problems/last-day-where-you-can-still-cross/solutions/3698920/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/30062023-1970-last-day-where-you?sd=pf)
![image.png](https://assets.leetcode.com/users/images/f43ca966-17b5-45f7-90dc-34e8a215fb95_1688137691.7631874.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/261
#### Problem TLDR
Last `day` matrix connected top-bottom when flooded each day at `cells[day]`
#### Intuition
One possible solution is to do a Binary Search in a days space, however it gives TLE.
Let's invert the problem: find the first day from the end where there is a connection top-bottom.
![image.png](https://assets.leetcode.com/users/images/db78d854-cdf6-489d-beee-5503db679ce5_1688100909.0815942.png)
Now, `cells[day]` is a new ground. We can use Union-Find to connect ground cells.

#### Approach
* use sentinel cells for `top` and `bottom`
* use path compressing `uf[n] = x`

#### Complexity

- Time complexity:
$$O(an)$$, where `a` is a reverse Ackerman function

- Space complexity:
$$O(n)$$

#### Code

```kotlin

val uf = HashMap<Int, Int>()
fun root(x: Int): Int = if (uf[x] == null || uf[x] == x) x else root(uf[x]!!)
.also { uf[x] = it }
fun latestDayToCross(row: Int, col: Int, cells: Array<IntArray>) =
    cells.size - 1 - cells.reversed().indexOfFirst { (y, x) ->
        uf[y * col + x] = root(if (y == 1) 0 else if (y == row) 1 else y * col + x)
        sequenceOf(y to x - 1, y to x + 1, y - 1 to x, y + 1 to x)
        .filter { (y, x) -> y in 1..row && x in 1..col }
        .map { (y, x) -> y * col + x }
        .forEach { if (uf[it] != null) uf[root(y * col + x)] = root(it) }
        root(0) == root(1)
    }

```

# 29.06.2023
[864. Shortest Path to Get All Keys](https://leetcode.com/problems/shortest-path-to-get-all-keys/description/) hard
[blog post](https://leetcode.com/problems/shortest-path-to-get-all-keys/solutions/3695847/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/29062023-864-shortest-path-to-get?sd=pf)
![image.png](https://assets.leetcode.com/users/images/1052b0c3-e4eb-458d-a74c-7e943089088d_1688026890.2702477.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/260
#### Problem TLDR
Min steps to collect all `lowercase` keys in matrix. `#` and `uppercase` locks are blockers.
#### Intuition
What will not work:
* dynamic programming – gives TLE
* DFS – as we can visit cells several times

For the shortest path, we can make a Breadth-First Search wave in a space of the current position and collected keys set.

#### Approach
* Let's use bit mask for collected keys set
* all bits set are `(1 << countKeys) - 1`

#### Complexity

- Time complexity:
$$O(nm2^k)$$

- Space complexity:
$$O(nm2^k)$$

#### Code

```kotlin

val dir = arrayOf(0, 1, 0, -1)
data class Step(val y: Int, val x: Int, val keys: Int)
fun shortestPathAllKeys(grid: Array<String>): Int {
    val w = grid[0].length
    val s = (0..grid.size * w).first { '@' == grid[it / w][it % w] }
    val bit: (Char) -> Int = { 1 shl (it.toLowerCase().toInt() - 'a'.toInt()) }
    val visited = HashSet<Step>()
        val allKeys = (1 shl (grid.map { it.count { it.isLowerCase() } }.sum()!!)) - 1
        var steps = -1
        return with(ArrayDeque<Step>()) {
            add(Step(s / w, s % w, 0))
            while (isNotEmpty() && steps++ < grid.size * w) {
                repeat(size) {
                    val step = poll()
                    val (y, x, keys) = step
                    if (keys == allKeys) return steps - 1
                    if (x in 0 until w && y in 0..grid.lastIndex && visited.add(step)) {
                        val cell = grid[y][x]
                        if (cell != '#' && !(cell.isUpperCase() && 0 == (keys and bit(cell)))) {
                            val newKeys = if (cell.isLowerCase()) (keys or bit(cell)) else keys
                            var dx = -1
                            dir.forEach { dy ->  add(Step(y + dy, x + dx, newKeys)).also { dx = dy } }
                        }
                    }
                }
            }
            -1
        }
    }

```

# 28.06.2023
[1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/description/) medium
[blog post](https://leetcode.com/problems/path-with-maximum-probability/solutions/3691288/kotlin-dijkstra/)
[substack](https://dmitriisamoilenko.substack.com/p/28062023-1514-path-with-maximum-probability?sd=pf)
![image.png](https://assets.leetcode.com/users/images/143073aa-364e-495a-9787-d50e5e1c55b0_1687926999.9465983.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/259
#### Problem TLDR
Max probability path from `start` to `end` in a probability edges graph
#### Intuition
What didn't work:
* naive BFS, DFS with `visited` set - will not work, as we need to visit some nodes several times
* Floyd-Warshall - will solve this problem for every pair of nodes, but takes $$O(n^3)$$ and gives TLE
What will work: Dijkstra
#### Approach
* store probabilities from `start` to every node in an array
* the stop condition will be when there is no any `better` path

#### Complexity

- Time complexity:
$$O(EV)$$

- Space complexity:
$$O(EV)$$

#### Code

```kotlin

fun maxProbability(n: Int, edges: Array<IntArray>, succProb: DoubleArray, start: Int, end: Int): Double {
    val pstart = Array(n) { 0.0 }
    val adj = mutableMapOf<Int, MutableList<Pair<Int, Double>>>()
    edges.forEachIndexed { i, (from, to) ->
        adj.getOrPut(from) { mutableListOf() } += to to succProb[i]
        adj.getOrPut(to) { mutableListOf() } += from to succProb[i]
    }
    with(ArrayDeque<Pair<Int, Double>>()) {
        add(start to 1.0)
        while(isNotEmpty()) {
            val (curr, p) = poll()
            if (p <= pstart[curr]) continue
            pstart[curr] = p
            adj[curr]?.forEach { (next, pnext) -> add(next to p * pnext) }
        }
    }

    return pstart[end]
}

```

# 27.06.2023
[373. Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/description/) medium
[blog post](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/solutions/3687668/kotlin-hard-dijkstra/)
[substack](https://dmitriisamoilenko.substack.com/p/27062023-373-find-k-pairs-with-smallest?sd=pf)
![image.png](https://assets.leetcode.com/users/images/7bbe783b-b8e3-419d-92f2-1f2e12dd99be_1687844671.548487.png)

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
![image.png](https://assets.leetcode.com/users/images/4a7d9e2b-dfa8-4466-83a6-93f370bb4c31_1687845288.8454409.png)
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

# 26.06.2023
[2462. Total Cost to Hire K Workers](https://leetcode.com/problems/total-cost-to-hire-k-workers/description/) medium
[blog post](https://leetcode.com/problems/total-cost-to-hire-k-workers/solutions/3683531/kotlin-two-pointer-priority-queue/)
[substack](https://dmitriisamoilenko.substack.com/p/26062023-2462-total-cost-to-hire?sd=pf)
![image.png](https://assets.leetcode.com/users/images/f654304d-c58d-436f-a2f7-e995d6a1a832_1687756508.645389.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/257
#### Problem TLDR
The sum of the smallest cost from suffix and prefix of a `costs` size of `candidates` in `k` iterations
#### Intuition
Description of the problem is rather ambiguous: we actually need to consider `candidates` count of items from the head and from the tail of the `costs` array. Then we can use `PriorityQueue` to choose the minimum and adjust two pointers `lo` and `hi`.

#### Approach
* use separate condition, when `2 * candidates >= costs.size`
* careful with indexes, check yourself by doing dry run
* we can use separate variable `takenL` and `takenR` or just use queue's sizes to minify the code

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

        fun totalCost(costs: IntArray, k: Int, candidates: Int): Long {
            val pqL = PriorityQueue<Int>()
            val pqR = PriorityQueue<Int>()
            var lo = 0
            var hi = costs.lastIndex
            var sum = 0L
            var count = 0
            if (2 * candidates >= costs.size) while (lo <= hi) pqL.add(costs[lo++])
            while (pqL.size < candidates && lo <= hi) pqL.add(costs[lo++])
            while (pqR.size < candidates && lo < hi) pqR.add(costs[hi--])
            while (lo <= hi && count++ < k) {
                if (pqR.peek() < pqL.peek()) {
                    sum += pqR.poll()
                    pqR.add(costs[hi--])
                } else {
                    sum += pqL.poll()
                    pqL.add(costs[lo++])
                }
            }
            while (pqR.isNotEmpty()) pqL.add(pqR.poll())
            while (count++ < k && pqL.isNotEmpty()) sum += pqL.poll()
            return sum
        }

```

# 25.06.2023
[1575. Count All Possible Routes](https://leetcode.com/problems/count-all-possible-routes/description/) hard
[blog post](https://leetcode.com/problems/count-all-possible-routes/solutions/3679289/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/25062023-1575-count-all-possible?sd=pf)
![image.png](https://assets.leetcode.com/users/images/1cbba751-8fac-4b42-a767-6b43454ddb66_1687665781.8780994.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/256
#### Problem TLDR
Count paths from `start` to `finish` using `|locations[i]-locations[j]` of the `fuel`
#### Intuition
Let's observe the example:

```

//  0 1 2 3 4
//  2 3 6 8 4
//    *   *
//
//  2 3 4 6 8
//    *     *
//
//  3-2(4)-3(3)-6(0)
//  3-6(2)-8(0)
//  3-8(5)
//  3-8(5)-6(3)-8(1)
//  3-4(4)-6(2)-8(0)

```

At each position `curr` given the amount of fuel `f` there is a certain number of ways to `finish`. It is independent of all the other factors, so can be safely cached.
#### Approach
* as there are also paths from `finish` to `finish`, modify the code to search other paths when `finish` is reached

#### Complexity

- Time complexity:
$$O(nf)$$, `f` - is a max fuel

- Space complexity:
$$O(nf)$$

#### Code

```kotlin

fun countRoutes(locations: IntArray, start: Int, finish: Int, fuel: Int): Int {
    //  0 1 2 3 4
    //  2 3 6 8 4
    //    *   *
    //
    //  2 3 4 6 8
    //    *     *
    //
    //  3-2(4)-3(3)-6(0)
    //  3-6(2)-8(0)
    //  3-8(5)
    //  3-8(5)-6(3)-8(1)
    //  3-4(4)-6(2)-8(0)

    val cache = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(curr: Int, f: Int): Int {
        if (f < 0) return 0
        return cache.getOrPut(curr to f) {
            var sum = if (curr == finish) 1 else 0
            locations.forEachIndexed { i, n ->
                if (i != curr) {
                    sum = (sum + dfs(i, f - Math.abs(n - locations[curr]))) % 1_000_000_007
                }
            }
            return@getOrPut sum
        }
    }
    return dfs(start, fuel)
}

```

# 24.06.2023
[956. Tallest Billboard](https://leetcode.com/problems/tallest-billboard/description/) hard
[blog post](https://leetcode.com/problems/tallest-billboard/solutions/3675652/kotlin-dfs-memo-hard-trick/)
[substack](https://dmitriisamoilenko.substack.com/p/24062023-956-tallest-billboard?sd=pf)
![image.png](https://assets.leetcode.com/users/images/4dc1d8cc-8ce9-4051-a074-d5d55a65e5e0_1687581919.7716997.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/255
#### Problem TLDR
Max sum of disjoint set in array
#### Intuition
Naive Dynamic Programming solution is to do a full search, adding to the first and to the second sums. That will give Out of Memory for this problem constraints.

```

dp[i][firstSum][secondSum] -> Out of Memory

```

The trick to make it work and consume less memory, is to cache only the difference `firstSum - secondSum`. It will slightly modify the code, but the principle is the same: try to add to the first, then to the second, otherwise skip.

#### Approach
* we can compute the first sum, as when `diff == 0` then `sum1 == sum2`

#### Complexity

- Time complexity:
$$O(nm)$$, `m` is a max difference

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

fun tallestBillboard(rods: IntArray): Int {
    val cache = Array(rods.size + 1) { Array(10000) { -1 } }
    fun dfs(curr: Int, sumDiff: Int): Int {
        if (curr == rods.size) return if (sumDiff == 0) 0 else Int.MIN_VALUE / 2

        return cache[curr][sumDiff + 5000].takeIf { it != -1 } ?: {
            val take1 = rods[curr] + dfs(curr + 1, sumDiff + rods[curr])
            val take2 = dfs(curr + 1, sumDiff - rods[curr])
            val notTake = dfs(curr + 1, sumDiff)
            maxOf(take1, take2, notTake)
        }().also { cache[curr][sumDiff + 5000] = it }
    }
    return dfs(0, 0)
}

```

# 23.06.2023
[1027. Longest Arithmetic Subsequence](https://leetcode.com/problems/longest-arithmetic-subsequence/description/) medium
[blog post](https://leetcode.com/problems/longest-arithmetic-subsequence/solutions/3673731/kotlin-hard-problem-n-3/)
[substack](https://dmitriisamoilenko.substack.com/p/23062023-1027-longest-arithmetic?sd=pf)
![image.png](https://assets.leetcode.com/users/images/ab4900f5-cc42-4f61-a18e-d727a626531f_1687526476.9427776.png)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/254

#### Problem TLDR
Max arithmetic subsequence length in array
#### Intuition
This was a hard problem for me :)
Naive Dynamic Programming solution with recursion and cache will give TLE.
Let's observe the result, adding numbers one-by-one:

```

// 20 1 15 3 10 5 8
// 20
// 20 1
//  1
// 20 20  1 15
//  1 15 15
//
// 20 20 20  1 1 15 3
// 1  15  3 15 3 3
//
// 20 20 20 20  1 1  1 15 15 10
//  1 15  3 10 15 3 10  3 10
//    10
//
// 20 20 20 20 20  1 1  1 1 15 15 15 10 5
//  1 15  3 10  5 15 3 10 5  3 10  5  5
//    10                        5
//     5
//
// 20 20 20 20 20 20  1 1  1 1 1 15 15 15 15 10 10 5 8
//  1 15  3 10  5  8 15 3 10 5 8  3 10  5  8  5  8 8
//    10                             5

```

For each pair `from-to` there is a sequence. When adding another number, we know what `next` numbers are expected.

#### Approach
We can put those sequences in a `HashMap` by `next` number key.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

data class R(var next: Int, val d: Int, var size: Int)
fun longestArithSeqLength(nums: IntArray): Int {
    // 20 1 15 3 10 5 8
    // 20
    // 20 1
    //  1
    // 20 20  1 15
    //  1 15 15
    //
    // 20 20 20  1 1 15 3
    // 1  15  3 15 3 3
    //
    // 20 20 20 20  1 1  1 15 15 10
    //  1 15  3 10 15 3 10  3 10
    //    10
    //
    // 20 20 20 20 20  1 1  1 1 15 15 15 10 5
    //  1 15  3 10  5 15 3 10 5  3 10  5  5
    //    10                        5
    //     5
    //
    // 20 20 20 20 20 20  1 1  1 1 1 15 15 15 15 10 10 5 8
    //  1 15  3 10  5  8 15 3 10 5 8  3 10  5  8  5  8 8
    //    10                             5

    val nextToR = mutableMapOf<Int, MutableList<R>>()
        var max = 2
        nums.forEachIndexed { to, num ->
            nextToR.remove(num)?.forEach { r ->
                r.next = num + r.d
                max = maxOf(max, ++r.size)
                nextToR.getOrPut(r.next) { mutableListOf() } += r
            }
            for (from in 0..to - 1) {
                val d = num - nums[from]
                val next = num + d
                nextToR.getOrPut(next) { mutableListOf() } += R(next, d, 2)
            }
        }
        return max
    }

```

# 22.06.2023
[714. Best Time to Buy and Sell Stock with Transaction Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/) medium
[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/solutions/3668167/kotlin-track-money-balance/)
[substack](https://dmitriisamoilenko.substack.com/p/22062023-714-best-time-to-buy-and?sd=pf)
![image.png](https://assets.leetcode.com/users/images/959be11c-5f02-4f48-af37-27de10a2a9a4_1687414937.016538.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/253
#### Problem TLDR
Max profit from buying stocks and selling them with `fee` for `prices[day]`
#### Intuition
Naive recursive or iterative Dynamic Programming solution will take $$O(n^2)$$ time if we iterate over all days for buying and for selling.
The trick here is to consider the money balances you have each day. We can track two separate money balances: for when we're buying the stock `balanceBuy` and for when we're selling `balanceSell`. Then, it is simple to greedily track balances:
* if we choose to buy, we subtract `prices[day]` from `balanceBuy`
* if we choose to sell, we add `prices[day] - fee` to `balanceSell`
* just greedily compare previous balances with choices and choose maximum balance.

#### Approach
* balances are always following each other: `buy-sell-buy-sell..`, or we can rewrite this like `currentBalance = maxOf(balanceSell, balanceBuy)` and use it for addition and subtraction.
* we can keep only the previous balances, saving space to $$O(1)$$
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun maxProfit(prices: IntArray, fee: Int) = prices
.fold(-prices[0] to 0) { (balanceBuy, balance), price ->
    maxOf(balanceBuy, balance - price) to maxOf(balance, balanceBuy + price - fee)
}.second

```

# 21.06.2023
[2448. Minimum Cost to Make Array Equal](https://leetcode.com/problems/minimum-cost-to-make-array-equal/description/) hard
[blog post](https://leetcode.com/problems/minimum-cost-to-make-array-equal/solutions/3663809/kotlin-binary-search/)
[substack](https://dmitriisamoilenko.substack.com/p/21062023-2448-minimum-cost-to-make?sd=pf)
![image.png](https://assets.leetcode.com/users/images/d8efb32b-45b9-45d6-806e-fc0d5f788db5_1687316516.3453755.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/252
#### Problem TLDR
Min cost to make all `arr[i]` equal, where each change is `cost[i]`
#### Intuition
First idea is that at least one element can be unchanged.
Assume, that we want to keep the most costly element unchanged, but this will break on example:

```

1 2 2 2    2 1 1 1
f(1) = 0 + 1 + 1 + 1 = 3
f(2) = 2 + 0 + 0 + 0 = 2 <-- more optimal

```

Let's observe the resulting cost for each number:

```

//    1 2 3 2 1     2 1 1 1 1
//0:  2 2 3 2 1 = 10
//1:  0 1 2 1 0 = 4
//2:  2 0 1 0 1 = 4
//3:  4 1 0 1 2 = 8
//4:  6 2 1 2 3 = 14

```

We can see that `f(x)` have a minimum and is continuous. We can find it with Binary Search, comparing the `slope = f(mid + 1) - f(mid - 1)`. If `slope > 0`, minimum is on the left.

#### Approach
For more robust Binary Search:
* use inclusive `lo`, `hi`
* always compute the result `min`
* always move the borders `lo = mid + 1` or `hi = mid - 1`
* check the last case `lo == hi`

#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun minCost(nums: IntArray, cost: IntArray): Long {
    //    1 2 3 2 1     2 1 1 1 1
    //0:  2 2 3 2 1 = 10
    //1:  0 1 2 1 0 = 4
    //2:  2 0 1 0 1 = 4
    //3:  4 1 0 1 2 = 8
    //4:  6 2 1 2 3 = 14
    fun costTo(x: Long): Long {
        return nums.indices.map { Math.abs(nums[it].toLong() - x) * cost[it].toLong() }.sum()
    }
    var lo = nums.min()?.toLong() ?: 0L
    var hi = nums.max()?.toLong() ?: 0L
    var min = costTo(lo)
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        val costMid1 = costTo(mid - 1)
        val costMid2 = costTo(mid + 1)
        min = minOf(min, costMid1, costMid2)
        if (costMid1 < costMid2) hi = mid - 1 else lo = mid + 1
    }
    return min
}

```

# 20.06.2023
[2090. K Radius Subarray Averages](https://leetcode.com/problems/k-radius-subarray-averages/description/) medium
[blog post](https://leetcode.com/problems/k-radius-subarray-averages/solutions/3659377/kotlin-sliding-window/)
[substack](https://dmitriisamoilenko.substack.com/p/20062023-2090-k-radius-subarray-averages?sd=pf)
![image.png](https://assets.leetcode.com/users/images/dcffd4a7-b3f7-4697-8b66-4c13a15e689c_1687231777.9171236.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/251
#### Problem TLDR
Array containing sliding window of size `2k+1` average or `-1`
#### Intuition
Just do what is asked

#### Approach
* careful with `Int` overflow
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun getAverages(nums: IntArray, k: Int): IntArray {
    if (k == 0) return nums
    var sum = 0L
    val res = IntArray(nums.size) { -1 }
    for (i in 0 until nums.size) {
        sum += nums[i]
        if (i > 2 * k) sum -= nums[i - 2 * k - 1]
        if (i >= 2 * k) res[i - k] = (sum / (2 * k + 1)).toInt()
    }
    return res
}

```

# 19.06.2023
[1732. Find the Highest Altitude](https://leetcode.com/problems/find-the-highest-altitude/description/) easy
[blog post](https://leetcode.com/problems/find-the-highest-altitude/solutions/3654634/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/18062023-1732-find-the-highest-altitude?sd=pf)
![image.png](https://assets.leetcode.com/users/images/257d1f08-62ae-49cf-84d5-304d497c79dd_1687146109.8661134.png)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/250
#### Problem TLDR
Max running sum
#### Intuition
Just sum all the values and compute the `max`

#### Approach
Let's write Kotlin `fold` one-liner
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun largestAltitude(gain: IntArray): Int = gain
.fold(0 to 0) { (max, sum), t -> maxOf(max, sum + t) to (sum + t) }
.first

```

# 18.06.2023
[2328. Number of Increasing Paths in a Grid](https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/description/) hard
[blog post](https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/solutions/3651039/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/18062023-2328-number-of-increasing?sd=pf)
![image.png](https://assets.leetcode.com/users/images/f702bc6b-8491-46ce-a067-5c15162c763f_1687066373.0051703.png)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/249
#### Problem TLDR
Count increasing paths in a matrix
#### Intuition
For every cell in a matrix, we can calculate how many increasing paths are starting from it. This result can be memorized. If we know the sibling's result, then we add it to the current if `curr > sibl`.

#### Approach
* use Depth-First search for the paths finding
* use `LongArray` for the memo
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun countPaths(grid: Array<IntArray>): Int {
    val m = 1_000_000_007L
    val counts = Array(grid.size) { LongArray(grid[0].size) }
    fun dfs(y: Int, x: Int): Long {
        return counts[y][x].takeIf { it != 0L } ?: {
            val v = grid[y][x]
            var sum = 1L
            if (x > 0 && v > grid[y][x - 1]) sum = (sum + dfs(y, x - 1)) % m
            if (y > 0 && v > grid[y - 1][x]) sum = (sum + dfs(y - 1, x)) % m
            if (y < grid.size - 1 && v > grid[y + 1][x]) sum = (sum + dfs(y + 1, x)) % m
            if (x < grid[0].size - 1 && v > grid[y][x + 1]) sum = (sum + dfs(y, x + 1)) % m
            sum
        }().also { counts[y][x] = it }
    }
    return (0 until grid.size * grid[0].size)
    .fold(0L) { r, t -> (r + dfs(t / grid[0].size, t % grid[0].size)) % m }
    .toInt()
}

```

# 17.06.2023
[1187. Make Array Strictly Increasing](https://leetcode.com/problems/make-array-strictly-increasing/description/) hard
[blog post](https://leetcode.com/problems/make-array-strictly-increasing/solutions/3647345/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/17062023-1187-make-array-strictly?sd=pf)

![image.png](https://assets.leetcode.com/users/images/f0efb026-48d7-4f89-9753-2ade3d32a976_1686985014.944183.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/248
#### Problem TLDR
Minimum replacements to make `arr1` increasing using any numbers `arr2`
#### Intuition
For any current position in `arr1` we can leave this number or replace it with any number from `arr2[i] > curr`. We can write Depth-First Search to check all possible replacements. To memorize, we must also consider the previous value. It can be used as-is, but more optimally, we just store a `skipped` boolean flag and restore the `prev` value: if it was skipped, then previous is from `arr1` else from `arr2`.

#### Approach
* sort and distinct the `arr2`
* use `Array` for cache, as it will be faster than a `HashMap`
* use explicit variable for the invalid result
* for the stop condition, if all the `arr1` passed, then result it good
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun makeArrayIncreasing(arr1: IntArray, arr2: IntArray): Int {
    val list2 = arr2.distinct().sorted()
    val INV = -1
    val cache = Array(arr1.size + 1) { Array(list2.size + 1) { IntArray(2) { -2 } } }
    fun dfs(pos1: Int, pos2: Int, skipped: Int): Int {
        val prev = if (skipped == 1) arr1.getOrNull(pos1-1)?:-1 else list2.getOrNull(pos2-1)?:-1
        return if (pos1 == arr1.size) 0 else cache[pos1][pos2][skipped].takeIf { it != -2} ?:
        if (pos2 == list2.size) {
            if (arr1[pos1] > prev) dfs(pos1 + 1, pos2, 1) else INV
        } else if (list2[pos2] <= prev) {
            dfs(pos1, pos2 + 1, 1)
        } else {
            val replace = dfs(pos1 + 1, pos2 + 1, 0)
            val skip = if (arr1[pos1] > prev) dfs(pos1 + 1, pos2, 1) else INV
            if (skip != INV && replace != INV) minOf(skip, 1 + replace)
            else if (replace != INV) 1 + replace else skip
        }.also { cache[pos1][pos2][skipped] = it }
    }
    return dfs(0, 0, 1)
}

```

# 16.06.2023
[1569. Number of Ways to Reorder Array to Get Same BST](https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/description/) hard
[blog post](https://leetcode.com/problems/number-of-ways-to-reorder-array-to-get-same-bst/solutions/3643907/kotlin-build-tree-count-permuts/)
[substack](https://dmitriisamoilenko.substack.com/p/16062023-1569-number-of-ways-to-reorder?sd=pf)
![image.png](https://assets.leetcode.com/users/images/19dcf6c9-718c-47b6-884b-4b4d474e027f_1686892068.5150518.png)

#### Join me on Telegram Leetcode_daily
https://t.me/leetcode_daily_unstoppable/247

#### Problem TLDR
Count permutations of an array with identical Binary Search Tree
#### Intuition
First step is to build a Binary Search Tree by adding the elements one by one.
Let's observe what enables the permutations in `[34512]`:
![image.png](https://assets.leetcode.com/users/images/4ab365a0-0ca6-4fe5-a25c-7fce3ee4b0e4_1686892282.3727634.png)
Left child `[12]` don't have permutations, as `1` must be followed by `2`. Same for the right `[45]`. However, when we're merging left and right, they can be merged in different positions.
Let's observe the pattern for merging `ab` x `cde`, `ab` x `cd`, `ab` x `c`, `a` x `b`:
![image.png](https://assets.leetcode.com/users/images/f4ddc8e4-83ab-4b59-9cb4-0bd8216bed05_1686892531.7714736.png)
And another, `abc` x `def`:
![image.png](https://assets.leetcode.com/users/images/bc09c45f-2d01-44ad-a0db-f8f3e0ee14a8_1686892570.416291.png)
For each `length` of a left `len1` and right `len2` subtree, we can derive the equation for permutations `p`:
$$
p(len1, len2) = p(len1 - 1, len2) + p(len1, len2 - 1)
$$
Also, when left or right subtree have several permutations, like `abc`, `acb`, `cab`, and right `def`, `dfe`, the result will be multiplied `3 x 2`.

#### Approach
Build the tree, then compute the `p = left.p * right.p * p(left.len, right.len)` in a DFS.
#### Complexity
- Time complexity:
$$O(n^2)$$, n for tree walk, and n^2 for `f`
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

class Node(val v: Int, var left: Node? = null, var right: Node? = null)
data class R(val perms: Long, val len: Long)
fun numOfWays(nums: IntArray): Int {
    val mod = 1_000_000_007L
    var root: Node? = null
    fun insert(n: Node?, v: Int): Node {
        if (n == null) return Node(v)
        if (v > n.v) n.right = insert(n.right, v)
        else n.left = insert(n.left, v)
        return n
    }
    nums.forEach { root = insert(root, it) }
    val cache = mutableMapOf<Pair<Long, Long>, Long>()
    fun f(a: Long, b: Long): Long {
        return if (a < b) f(b, a) else if (a <= 0 || b <= 0) 1 else cache.getOrPut(a to b) {
            (f(a - 1, b) + f(a, b - 1)) % mod
        }
    }
    fun perms(a: R, b: R): Long {
        val perms = (a.perms * b.perms) % mod
        return (perms * f(a.len , b.len)) % mod
    }
    fun dfs(n: Node?): R {
        if (n == null) return R(1, 0)
        val left = dfs(n.left)
        val right = dfs(n.right)
        return R(perms(left, right), left.len + right.len + 1)
    }
    val res = dfs(root)?.perms?.dec() ?: 0
    return (if (res < 0) res + mod else res).toInt()
}

```

# 15.06.2023
[1161. Maximum Level Sum of a Binary Tree](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/solutions/3639491/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/15062023-1161-maximum-level-sum-of?sd=pf)
![image.png](https://assets.leetcode.com/users/images/0001f209-62ce-4c0e-9d35-921b77240056_1686800390.362235.png)

#### Join me on Telegram Leetcode_daily
https://t.me/leetcode_daily_unstoppable/246
#### Problem TLDR
Binary Tree level with max sum

#### Intuition
We can use Breadth-First Search to find a `sum` of each level.

#### Approach
Let's try to write it in a Kotlin style
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun maxLevelSum(root: TreeNode?) = with(ArrayDeque<TreeNode>()) {
    root?.let { add(it) }
    generateSequence<Int> {
        if (isEmpty()) null else (1..size).map {
            with(poll()) {
                `val`.also {
                    left?.let { add(it) }
                    right?.let { add(it) }
                }
            }
        }.sum()
    }.withIndex().maxBy { it.value }?.index?.inc() ?: 0
}

```

# 14.06.2023
[530. Minimum Absolute Difference in BST](https://leetcode.com/problems/minimum-absolute-difference-in-bst/description/) easy
[blog post](https://leetcode.com/problems/minimum-absolute-difference-in-bst/solutions/3635561/kotlin-morris-traversal/)
[substack](https://dmitriisamoilenko.substack.com/p/14062023-530-minimum-absolute-difference?sd=pf)
![image.png](https://assets.leetcode.com/users/images/6f699d52-79a0-4179-8b53-460e2a1842ce_1686714279.6705241.png)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/245
#### Problem TLDR
Min difference in a BST
#### Intuition
In-order traversal in a BST gives a sorted order, we can compare `curr - prev`.

#### Approach
Let's write a [Morris traversal](https://en.wikipedia.org/wiki/Threaded_binary_tree): make the current node a rightmost child of its left child.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun getMinimumDifference(root: TreeNode?): Int {
    if (root == null) return 0
    var minDiff = Int.MAX_VALUE
    var curr = root
    var prev = -1
    while (curr !=  null) {
        val left = curr.left
        if (left != null) {
            var leftRight = left
            while (leftRight.right != null) leftRight = leftRight.right
            leftRight.right = curr
            curr.left = null
            curr = left
        } else {
            if (prev >= 0) minDiff = minOf(minDiff, curr.`val` - prev)
            prev = curr.`val`
            curr = curr.right
        }
    }
    return minDiff
}

```

# 13.06.2023
[2352. Equal Row and Column Pairs](https://leetcode.com/problems/equal-row-and-column-pairs/description/) medium
[blog post](https://leetcode.com/problems/equal-row-and-column-pairs/solutions/3631323/kotlin-hash/)
[substack](https://dmitriisamoilenko.substack.com/p/12062023-2352-equal-row-and-column?sd=pf)
![image.png](https://assets.leetcode.com/users/images/ac9c8b85-0617-4b59-a269-f302ed1e3de3_1686628513.6269782.png)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/244
#### Problem TLDR
Count of `rowArray` == `colArray` in an `n x n` matrix.

#### Intuition
Compute `hash` function for each `row ` and each `col`, then compare them. If `hash(row) == hash(col)`, then compare arrays.
For hashing, we can use simple `31 * prev + curr`, that encodes both value and position.

#### Approach
* For this Leetcode data, `tan` hash works perfectly, we can skip comparing the arrays.

#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun equalPairs(grid: Array<IntArray>): Int {
    val rowHashes = grid.map { it.fold(0.0) { r, t ->  Math.tan(r) + t } }
    val colHashes = (0..grid.lastIndex).map { x ->
        (0..grid.lastIndex).fold(0.0) { r, t -> Math.tan(r) + grid[t][x] } }
        return (0..grid.size * grid.size - 1).count {
            rowHashes[it / grid.size] == colHashes[it % grid.size]
        }
    }

```

# 12.06.2023
![image.png](https://assets.leetcode.com/users/images/25c39272-e908-4b53-8202-06becd8adc74_1686541066.3963215.png)
[228. Summary Ranges](https://leetcode.com/problems/summary-ranges/description/) easy
[blog post](https://leetcode.com/problems/summary-ranges/solutions/3627478/kotlin-fold/)
[substack](https://dmitriisamoilenko.substack.com/p/12062023-228-summary-ranges?sd=pf)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/243
#### Problem TLDR
Fold continues ranges in a sorted array `1 2 3 5` -> `1->3, 5`
#### Intuition
Scan from start to end, modify the last interval or add a new one.

#### Approach
Let's write a Kotlin one-liner

#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun summaryRanges(nums: IntArray): List<String> = nums
    .fold(mutableListOf<IntArray>()) { r, t ->
        if (r.isEmpty() || r.last()[1] + 1 < t) r += intArrayOf(t, t)
        else r.last()[1] = t
        r
    }
    .map { (f, t) -> if (f == t) "$f" else "$f->$t"}

```

# 11.06.2023
[1146. Snapshot Array](https://leetcode.com/problems/snapshot-array/description/) medium
[blog post](https://leetcode.com/problems/snapshot-array/solutions/3623764/kotlin-binary-search/)
[substack](https://dmitriisamoilenko.substack.com/p/11062023-1146-snapshot-array?sd=pf)
![image.png](https://assets.leetcode.com/users/images/056cadfa-afa3-46c9-83ba-dd31277c0078_1686455279.7681677.png)

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

# 10.06.2023
[1802. Maximum Value at a Given Index in a Bounded Array](https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/description/) medium
[blog post](https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/solutions/3620296/kotlin-nums-i-can-t-be-zero/)
[substack](https://dmitriisamoilenko.substack.com/p/10062023-1802-maximum-value-at-a?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/241
#### Problem TLDR
Max at `index` in an `n` sized array, where `sum <= maxSum`, `nums[i] > 0` and `maxDiff(i, i+1) < 2`.
#### Intuition
Let's write possible numbers, for example:

```

// n=6, i=1, m=10
// 10/6 = 1
// 0 1 2 3 4 5
// -----------
// 0 1 0 0 0 0 sum = 1
// 1 2 1 0 0 0 sum = 1 + (1 + 1 + 1) = 4
// 2 3 2 1 0 0 sum = 4 + (1 + 2 + 1) = 8
// 3 4 3 2 1 0 sum = 8 + (1 + 3 + 1) = 13 > 10  prev + (1 + left + right)
// 4 5 4 3 2 1 sum = 13 + (1 + 4 + 1) = 19      left = minOf(left, i)
// 5 6 5 4 3 2 sum = 19 + (1 + 4 + 1) = 24      right = minOf(right, size - i - 1)
// 6 7 6 5 4 3
// ...
//   5+x       sum = 19 + x * (1 + 4 +1)
// ...
// S(x-1) - S(x-1-i) + x + S(x-1) - S(x-1 - (size-i-1))
// x + 2 * S(x-1) - S(x-1-i) - S(x-size+i)
// S(y) = y * (y + 1) / 2

```

We should minimize the sum for it to be `<= maxSum`, so naturally, we place the maximum at `index` and do strictly lower the sibling numbers.
Looking at the example, we see there is an arithmetic sum to the left and to the right of the `index`.
$$
S(n) = 1 + 2 + .. + (n-1) + n = n * (n+1) / 2
$$
We are also must subtract part of the sum, that out of the array:
$$
\sum = S(x-1) - S(x-1-i) + x + S(x-1) - S(x-1 - (size-i-1))
$$
Another catch, numbers can't be `0`, so we must start with an array filled of `1`: `1 1 1 1 1 1`. That will modify our algorithm, adding `n` to the `sum` and adding one more step to the `max`.

Given that we know `sum` for each `max`, and `sum` will grow with increasing of the `max`, we can do a binary search `sum = f(max)` for `max`.
#### Approach
For more robust binary search:
* use inclusive borders `lo` and `hi`
* check the last condition `lo == hi`
* always compute the result `max = mid`
* avoid the number overflow
#### Complexity
- Time complexity:
$$O(log(n))$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun maxValue(n: Int, index: Int, maxSum: Int): Int {

    val s: (Int) -> Long = { if (it < 0L) 0L else it.toLong() * (it.toLong() + 1L) / 2L }
    var lo = 0
    var hi = maxSum
    var max = lo
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        val sum = n + mid + 2L * s(mid - 1) - s(mid - 1 - index) - s(mid - n + index)
        if (sum <= maxSum) {
            max = mid
            lo = mid + 1
        } else hi = mid - 1
    }
    return max + 1
}

```

# 09.06.2023
[744. Find Smallest Letter Greater Than Target](https://leetcode.com/problems/find-smallest-letter-greater-than-target/) easy
[blog post](https://leetcode.com/problems/find-smallest-letter-greater-than-target/solutions/3616091/kotlin-binarysearch/)
[substack](https://dmitriisamoilenko.substack.com/p/09062023-744-find-smallest-letter?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/240
#### Problem TLDR
Lowest char greater than `target`.
#### Intuition
In a sorted array, we can use the Binary Search.

#### Approach
For more robust code:
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always move `lo` or `hi`
* always write a good result `res = ...`
* safely compute `mid`
#### Complexity
- Time complexity:
$$O(log(n))$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun nextGreatestLetter(letters: CharArray, target: Char): Char {
    var res = letters[0]
    var lo = 0
    var hi = letters.lastIndex
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (letters[mid] > target) {
            hi = mid - 1
            res = letters[mid]
        } else lo = mid + 1
    }
    return res
}

```

# 08.06.2023
[1351. Count Negative Numbers in a Sorted Matrix](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/description/) easy
[blog post](https://leetcode.com/problems/count-negative-numbers-in-a-sorted-matrix/solutions/3611472/kotlin-fold/)
[substack](https://dmitriisamoilenko.substack.com/p/08062023-1351-count-negative-numbers?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/239
#### Problem TLDR
Count negatives in a sorted by row and by column matrix.
#### Intuition
Consider example:

```

4  3  2 -1
3  2  1 -1
1  1 -1 -2
^ we are here
-1 -1 -2 -3

```

If we set position `x` at the first negative number, it is guaranteed, that the next `row[x]` will also be negative. So we can skip already passed columns.
#### Approach
Let's use Kotlin's `fold` operator.
#### Complexity
- Time complexity:
$$O(n + m)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun countNegatives(grid: Array<IntArray>): Int =
    grid.fold(0 to 0) { (total, prev), row ->
        var curr = prev
        while (curr < row.size && row[row.lastIndex - curr] < 0) curr++
        (total + curr) to curr
    }.first
}

```

# 07.06.2023
[1318. Minimum Flips to Make a OR b Equal to c](https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/description/) medium
[blog post](https://leetcode.com/problems/minimum-flips-to-make-a-or-b-equal-to-c/solutions/3607170/kotlin-or-and-xor/)
[substack](https://dmitriisamoilenko.substack.com/p/07062023-1318-minimum-flips-to-make?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/238
#### Problem TLDR
Minimum `a` and `b` Int bit flips to make `a or b == c`.
#### Intuition
Naive implementation is to iterate over `32` bits and flip `a` or/and `b` bits to match `c`.
If we didn't consider the case where `a = 1` and `b = 1` and `c = 0`, the result would be `(a or b) xor c`, as `a or b` gives us the left side of the equation, and `xor c` gives only bits that are needed to flip. For the corner case `a = b = 1, c = 0`, we must do additional flip to make `0`, and we must make any other combinations `0`:

```

a b c     a and b   c.inv()   (a and b) and c.inv()

0 0 1     0         0         0
0 1 0     0         1         0
0 1 1     0         0         0
1 0 0     0         1         0
1 0 1     0         0         0
1 1 0     1         1         1
1 1 1     1         0         0

```

#### Approach
Use `Integer.bitCount`.

#### Complexity
- Time complexity:
$$O(1)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun minFlips(a: Int, b: Int, c: Int): Int =
Integer.bitCount((a or b) xor c) + Integer.bitCount((a and b) and c.inv())

```

# 06.06.2023
[1502. Can Make Arithmetic Progression From Sequence](https://leetcode.com/problems/can-make-arithmetic-progression-from-sequence/description/) easy
[blog post](https://leetcode.com/problems/can-make-arithmetic-progression-from-sequence/solutions/3602840/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/06062023-1502-can-make-arithmetic?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/237
#### Problem TLDR
Is `IntArray` can be arithmetic progression?
#### Intuition
Sort, then use sliding window.

#### Approach
Let's write Kotlin one-liner.
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun canMakeArithmeticProgression(arr: IntArray): Boolean =
arr.sorted().windowed(2).groupBy { it[1] - it[0] }.keys.size == 1

```

# 05.06.2023
[1232. Check If It Is a Straight Line](https://leetcode.com/problems/check-if-it-is-a-straight-line/description/) easy
[blog post](https://leetcode.com/problems/check-if-it-is-a-straight-line/solutions/3598943/kotlin-tan/)
[substack](https://dmitriisamoilenko.substack.com/p/05062023-1232-check-if-it-is-a-straight?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/236
#### Problem TLDR
Are all the `x,y` points in a line?
#### Intuition
We can compare $$tan_i = dy_i/dx_i = dy_0/dx_0$$

#### Approach
* corner case is a vertical line
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun checkStraightLine(coordinates: Array<IntArray>): Boolean =
    with((coordinates[1][1] - coordinates[0][1])/
    (coordinates[1][0] - coordinates[0][0]).toDouble()) {
        coordinates.drop(2).all {
            val o = (it[1] - coordinates[0][1]) / (it[0] - coordinates[0][0]).toDouble()

            isInfinite() && o.isInfinite() || this == o
        }
    }

```

# 04.06.2023
[547. Number of Provinces](https://leetcode.com/problems/number-of-provinces/description/) medium
[blog post](https://leetcode.com/problems/number-of-provinces/solutions/3594857/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/04062023-547-number-of-provinces?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/235
#### Problem TLDR
Count connected groups in graph.
#### Intuition
Union-Find will perfectly fit to solve this problem.

#### Approach
For more optimal Union-Find:
* use path compression in the `root` method: `uf[it] = x`
* connect the smallest size subtree to the largest
#### Complexity
- Time complexity:
$$O(a(n)n^2)$$, `a(n)` - reverse Ackerman function `f(x) = 2^2^2..^2, x times`. `a(Int.MAX_VALUE) = 2^32 = 2^2^5 == 3`
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun findCircleNum(isConnected: Array<IntArray>): Int {
    val uf = IntArray(isConnected.size) { it }
    val sz = IntArray(isConnected.size) { 1 }
    var count = uf.size
    val root: (Int) -> Int = {
        var x = it
        while (uf[x] != x) x = uf[x]
        uf[it] = x
        x
    }
    val connect: (Int, Int) -> Unit = { a, b ->
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) {
            count--
            if (sz[rootA] < sz[rootB]) {
                uf[rootB] = rootA
                sz[rootA] += sz[rootB]
                sz[rootB] = 0
            } else {
                uf[rootA] = rootB
                sz[rootB] += sz[rootA]
                sz[rootA] = 0
            }
        }
    }
    for (i in 0..sz.lastIndex)
    for (j in 0..sz.lastIndex)
    if (isConnected[i][j] == 1) connect(i, j)
    return count
}

```

# 03.06.2023
[1376. Time Needed to Inform All Employees](https://leetcode.com/problems/time-needed-to-inform-all-employees/description/) medium
[blog post](https://leetcode.com/problems/time-needed-to-inform-all-employees/solutions/3591362/kotlin-dfs/)
[substack](https://dmitriisamoilenko.substack.com/p/03062023-1376-time-needed-to-inform?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/234
#### Problem TLDR
Total `time` from `headID` to all nodes in graph.
#### Intuition
Total time will be the maximum time from the root of the graph to the lowest node. To find it out, we can use DFS.
#### Approach
Build the graph, then write the DFS.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun numOfMinutes(n: Int, headID: Int, manager: IntArray, informTime: IntArray): Int {
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        (0 until n).forEach { fromTo.getOrPut(manager[it]) { mutableListOf() } += it }
        fun dfs(curr: Int): Int {
            return informTime[curr] + (fromTo[curr]?.map { dfs(it) }?.max() ?: 0)
        }
        return dfs(headID)
    }

```

# 02.06.2023
[2101. Detonate the Maximum Bombs](https://leetcode.com/problems/detonate-the-maximum-bombs/description/) medium
[blog post](https://leetcode.com/problems/detonate-the-maximum-bombs/solutions/3587925/kotlin-directed-graph/)
[substack](https://dmitriisamoilenko.substack.com/p/02062023-2101-detonate-the-maximum?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/233
#### Problem TLDR
Count detonated bombs by chain within each radius.
#### Intuition
A bomb will only detonate if its center within the radius of another.
![image.png](https://assets.leetcode.com/users/images/0a0ae515-9b35-44b0-9cb6-cd18b72980ca_1685679480.0795984.png)
For example, `A` can detonate `B`, but not otherwise.

Let's build a graph, who's who can detonate.
#### Approach
Build a graph, the do DFS trying to start from each node.
#### Complexity
- Time complexity:
$$O(n^3)$$, each of the `n` DFS will take $$n^2$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun maximumDetonation(bombs: Array<IntArray>): Int {
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        for (i in 0..bombs.lastIndex) {
            val bomb1 = bombs[i]
            val rr = bomb1[2] * bomb1[2].toLong()
            val edges = fromTo.getOrPut(i) { mutableListOf() }
            for (j in 0..bombs.lastIndex) {
                if (i == j) continue
                val bomb2 = bombs[j]
                val dx = (bomb1[0] - bomb2[0]).toLong()
                val dy = (bomb1[1] - bomb2[1]).toLong()
                if (dx * dx + dy * dy <= rr) edges += j
            }
        }
        fun dfs(curr: Int, visited: HashSet<Int> = HashSet()): Int {
            return if (visited.add(curr)) {
                1 + (fromTo[curr]?.sumBy { dfs(it, visited) } ?:0)
            } else 0
        }
        var max = 1
        for (i in 0..bombs.lastIndex) max = maxOf(max, dfs(i))
        return max
    }

```

# 01.06.2023
[1091. Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/description/) medium
[blog post](https://leetcode.com/problems/shortest-path-in-binary-matrix/solutions/3584350/kotln-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/01062023-1091-shortest-path-in-binary?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/232
#### Problem TLDR
`0` path length in a binary square matrix.
#### Intuition
Just do BFS.

#### Approach
Some tricks for cleaner code:
* check for x, y in `range`
* iterate over `dirs`. This is a sequence of `x` and `y`
* modify the input array. But don't do this in production code.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun shortestPathBinaryMatrix(grid: Array<IntArray>): Int =
    with(ArrayDeque<Pair<Int, Int>>()) {
        val range = 0..grid.lastIndex
        val dirs = arrayOf(0, 1, 0, -1, -1, 1, 1, -1)
        if (grid[0][0] == 0) add(0 to 0)
        grid[0][0] = -1
        var step = 0
        while (isNotEmpty()) {
            step++
            repeat(size) {
                val (x, y) = poll()
                if (x == grid.lastIndex && y == grid.lastIndex) return step
                var dx = -1
                for (dy in dirs) {
                    val nx = x + dx
                    val ny = y + dy
                    if (nx in range && ny in range && grid[ny][nx] == 0) {
                        grid[ny][nx] = -1
                        add(nx to ny)
                    }
                    dx = dy
                }
            }
        }
        -1
    }

```

# 31.05.2023
[1396. Design Underground System](https://leetcode.com/problems/design-underground-system/description/) medium
[blog post](https://leetcode.com/problems/design-underground-system/solutions/3580723/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/31052023-1396-design-underground?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/229
#### Problem TLDR
Average time `from, to` when different user IDs do `checkIn(from, time1)` and `checkOut(to, time2)`
#### Intuition
Just do what is asked, use `HashMap` to track user's last station.

#### Approach
* store `sum` time and `count` for every `from, to` station
* use `Pair` as key for `HashMap`
#### Complexity
- Time complexity:
$$O(1)$$, for each call
- Space complexity:
$$O(n)$$

#### Code

```kotlin

class UndergroundSystem() {
    val fromToSumTime = mutableMapOf<Pair<String, String>, Long>()
    val fromToCount = mutableMapOf<Pair<String, String>, Int>()
    val idFromTime = mutableMapOf<Int, Pair<String, Int>>()
    fun Pair<String, String>.time() = fromToSumTime[this] ?: 0L
    fun Pair<String, String>.count() = fromToCount[this] ?: 0

    fun checkIn(id: Int, stationName: String, t: Int) {
        idFromTime[id] = stationName to t
    }

    fun checkOut(id: Int, stationName: String, t: Int) {
        val (from, tFrom) = idFromTime[id]!!
        val fromTo = from to stationName
        fromToSumTime[fromTo] = t - tFrom + fromTo.time()
        fromToCount[fromTo] = 1 + fromTo.count()
    }

    fun getAverageTime(startStation: String, endStation: String): Double =
    with(startStation to endStation) {
        time().toDouble() / count().toDouble()
    }

}

```

# 30.05.2023
[705. Design HashSet](https://leetcode.com/problems/design-hashset/description/) easy
[blog post](https://leetcode.com/problems/design-hashset/solutions/3577326/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/28052023-705-design-hashset?sd=pf)
#### Telegram
https://t.me/leetcode_daily_unstoppable/228
#### Problem TLDR
Write a `HashSet`.
#### Intuition
There are different [hash functions](https://en.wikipedia.org/wiki/Hash_function). Interesting implementations is In Java `HashMap` [https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/HashMap.java](https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/HashMap.java)

#### Approach
Use `key % size` for the hash function, grow and rehash when needed.

#### Complexity
- Time complexity:
$$O(1)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

class MyHashSet(val initialSz: Int = 16, val loadFactor: Double = 1.6) {
            var buckets = Array<LinkedList<Int>?>(initialSz) { null }
            var size = 0

            fun hash(key: Int): Int = key % buckets.size

            fun rehash() {
            if (size > buckets.size * loadFactor) {
                val oldBuckets = buckets
                buckets = Array<LinkedList<Int>?>(buckets.size * 2) { null }
                    oldBuckets.forEach { it?.forEach { add(it) } }
                }
            }

            fun bucket(key: Int): LinkedList<Int> {
                val hash = hash(key)
                if (buckets[hash] == null) buckets[hash] = LinkedList<Int>()
                    return buckets[hash]!!
                }

                fun add(key: Int) {
                    val list = bucket(key)
                    if (!list.contains(key)) {
                        list.add(key)
                        size++
                        rehash()
                    }
                }

                fun remove(key: Int) {
                    bucket(key).remove(key)
                }

                fun contains(key: Int): Boolean =
                   bucket(key).contains(key)
}

```

# 29.05.2023
[1603. Design Parking System](https://leetcode.com/problems/design-parking-system/description/) easy
[blog post](https://leetcode.com/problems/design-parking-system/solutions/3573683/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/27052023-1603-design-parking-system?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/227
#### Problem TLDR
Return if car of type `1, 2 or 3` can be added given sizes `big, medium and small`.
#### Intuition
Just write the code.

#### Approach
Let's use an array to minimize the number of lines.
#### Complexity
- Time complexity:
$$O(1)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

class ParkingSystem(big: Int, medium: Int, small: Int) {
    val types = arrayOf(big, medium, small)

    fun addCar(carType: Int): Boolean = types[carType - 1]-- > 0
}

```

# 28.05.2023
[1547. Minimum Cost to Cut a Stick](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/submissions/958762191/) hard
[blog post](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/solutions/3570530/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/28052023-1547-minimum-cost-to-cut?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/226
#### Problem TLDR
Min cost of cuts `c1,..,ci,..,cn` of `[0..n]` where cut cost the `length = to-from`.
#### Intuition
We every stick `from..to` we can try all the cuts in that range. This result will be optimal and can be cached.

#### Approach
* use DFS + memo
* check for range
#### Complexity
- Time complexity:
$$k^2$$, as maximum depth of DFS is `k`, and we loop for `k`.
- Space complexity:
$$k^2$$

#### Code

```kotlin

fun minCost(n: Int, cuts: IntArray): Int {
    val cache = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(from: Int, to: Int): Int {
        return cache.getOrPut(from to to) {
            var min = 0
            cuts.forEach {
                if (it in from + 1..to - 1) {
                    val new = to - from + dfs(from, it) + dfs(it, to)
                    if (min == 0 || new < min) min = new
                }
            }

            min
        }
    }
    return dfs(0, n)
}

```

# 27.05.2023
[1406. Stone Game III](https://leetcode.com/problems/stone-game-iii/description/) hard
[blog post](https://leetcode.com/problems/stone-game-iii/solutions/3566578/kotln-dp-prefix-sum/)
[substack](https://dmitriisamoilenko.substack.com/p/27052023-1406-stone-game-iii?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/225
#### Problem TLDR
Winner of “Alice”, “Bob” or “Tie” in game of taking `1, 2 or 3` stones by turn from `stoneValue` array.
#### Intuition
Let's count the result for Alice, starting at `i` element:
$$
alice_i = \sum_{i=1,2,3}(stones_i) + suffix_i - alice_{i+1}
$$
The result can be safely cached. Bob's points will be Alice's points in the next turn.
#### Approach
Let's write bottom up DP.
* use increased sizes for `cache` and `suffix` arrays for simpler code
* corner case is the negative number, so we must take at least one stone
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun stoneGameIII(stoneValue: IntArray): String {
    val suffix = IntArray(stoneValue.size + 1)
    for (i in stoneValue.lastIndex downTo 0) suffix[i] = stoneValue[i] + suffix[i + 1]
    val cache = IntArray(stoneValue.size + 1)
    var bob = 0

    for (curr in stoneValue.lastIndex downTo 0) {
        var sum = 0
        var first = true
        for (j in 0..2) {
            val ind = curr + j
            if (ind > stoneValue.lastIndex) break
            sum += stoneValue[ind]
            val nextAlice = cache[ind + 1]
            val next = suffix[ind + 1] - nextAlice
            if (first || sum + next > cache[curr]) {
                first = false
                cache[curr] = sum + next
                bob = nextAlice
            }
        }
    }
    return if (cache[0] == bob) "Tie" else if (cache[0] > bob) "Alice" else "Bob"
}

```

# 26.05.2023
[1140. Stone Game II](https://leetcode.com/problems/stone-game-ii/description/) medium
[blog post](https://leetcode.com/problems/stone-game-ii/solutions/3563651/kotlin-dfs-cache-prefix-sum/)
[substack](https://dmitriisamoilenko.substack.com/p/26052023-1140-stone-game-ii?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/224
#### Problem TLDR
While Alice and Bob optimally take `1..2*m` numbers from `piles` find maximum for Alice.
#### Intuition
For each position, we can cache the result for Alice starting from it. Next round, Bob will become Alice and use that cached result, but Alice will use the remaining part:
$$
bob_i = suffix_i - alice_i
$$
and
$$
alice_i = \sum(piles_{1..x<2m}) + (suffix_i - alice_{i + 1})
$$

#### Approach
* compute prefix sums in `IntArray`
* use `HashMap` for simpler code, or Array for faster
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun stoneGameII(piles: IntArray): Int {
    // 2 7 9 4 4    M      A   B
    // A            1      1
    // A A          2      2,7
    // A B B        2          7,9
    // A A B B B    3          9,4,4
    val sums = IntArray(piles.size)
    sums[0] = piles[0]
    for (i in 1..piles.lastIndex)
    sums[i] = sums[i - 1] + piles[i]
    val total = sums[sums.lastIndex]
    val cache = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(m: Int, curr: Int): Int {
        return cache.getOrPut(m to curr) {
            var res = 0
            var pilesSum = 0
            for (x in 0 until 2*m) {
                if (curr + x > piles.lastIndex) break
                pilesSum += piles[curr + x]
                val nextOther = dfs(maxOf(m, x + 1), curr + x + 1)
                val nextMy = total - sums[curr + x] - nextOther
                res = maxOf(res, pilesSum + nextMy)
            }
            res
        }
    }
    return dfs(1, 0)
}

```

# 25.05.2023
[837. New 21 Game](https://leetcode.com/problems/new-21-game/description/) medium
[blog post](https://leetcode.com/problems/new-21-game/solutions/3560756/kotlin-observe-and-simulate-then-math/)
[substack](https://dmitriisamoilenko.substack.com/p/25052023-837-new-21-game?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/223
#### Problem TLDR
Probability sum of random numbers `1..maxPts` sum be `< n` after it overflow `k`.
#### Intuition
For every event, we choose `one` number from numbers `1..maxPts`. Probability of this event is `p1 = 1/maxPts`.

For example, `n=6, k=1, maxpts=10`: we can pick any numbers `1, 2, 3, 4, 5, 6` that are `<=6`. Numbers `7, 8, 9, 10` are excluded, because they are `>6`. After we pick one number with probability `p1 = 1/10`, the sum will be `>=k` so we stop. The final probability is the sum of individual valid choices `p = sum(good_p1)`

Another example, `n=6, k=2, maxpts=10`: our choices are

```

// n = 6, k = 2, maxpts = 10
// p_win1 1+1, 1+2, 1+3, 1+4, 1+5, 2,   3,  4,  5,  6
//        0.01 0.01 0.01 0.01 0.01 0.1 0.1 0.1 0.1 0.1 = 0.55

```

When we go to the second round in cases of `1+1, 1+2, 1+3, 1+4, 1+5`, we multiply the probabilities, so `p(1+1) = p1*p1`.

Next, observe the pattern for other examples:

```

// n = 6, k = 3, maxpts = 10
// p_win  1+1+1, 1+1+2, 1+1+3, 1+1+4, 1+2, 1+3, 1+4, 1+5, 2+1, 2+2, 2+3, 2+4, 3,  4,  5,   6
//        0.001  0.001  0.001  0.001  0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.1 0.1 0.1 0.1
// sum=0.484

// n = 6, k = 4, maxpts = 10
// p_win  1+1+1+1 1+1+1+2 1+1+1+3 1+1+2 1+1+3 1+1+4 1+2+1 1+2+2 1+2+3 1+3  1+4  1+5  2+1+1 2+1+2 2+1+3 2+2  2+3  2+4  3+1  3+2  3+3  4   5   6
//         .0001   .0001   .0001   .001  .001  .001  .001  .001  .001  .01  .01  .01  .001  .001  .001  .01  .01  .01  .01  .01  .01  .1  .1  .1
//sum=0.3993

```

What we see is the sequence of `1+1+1+1, 1+1+1+2, 1+1+1+3`, where we pick a number from `1..maxpts` then calculate the sum and if the sum is still smaller than `n` we go deeper and make another choice from `1..maxpts`.
That can be written as Depth-First Search algorithm:

```

fun dfs(currSum: Int): Double {
    ...
    var sumP = 0.0
    for (x in 1..maxPts)
    sumP += dfs(currSum + x)
    res = sumP * p1
}

```

This will work and gives us correct answers, but gives TLE for big numbers, as its time complexity is $$O(n^2)$$.

Let's observe this algorithm's recurrent equation:
$$
f(x) = (f(x+1) + f(x+2)+..+f(x + maxPts))*p1
$$
$$
f(x + 1) = (f(x+2) + f(x+3) +...+f(x + maxPts)*p1 + f(x + 1 + maxPts))*p1
$$
subtract second sequence from the first:
$$
f(x) - f(x + 1) = f(x+1)*p1 - f(x+1+maxPts)*p1
$$
$$
f(x) = f(x+1) + (f(x+1) - f(x+1+maxPts))*p1
$$
This removes one dimension of iteration `1..maxPts`. However, it fails the first case where `currSum == k - 1`, because the equation didn't consider that not all `x+maxPts` are smaller than `n`. For this case, we must return `(n-k+1)*p1` as we choose last number from range `k - 1..n`.
#### Approach
This problem is next to impossible to solve in an interview, given how many conclusions you must derive on the fly. So, hope no one will give it to you.
* use an array for the faster cache
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun new21Game(n: Int, k: Int, maxPts: Int): Double {
    // n = 6, k = 1, maxpts = 10
    // cards: 1 2 3 4 5 6 7 8 9 10
    // p_win1(6, 10) = count(1 2 3 4 5 6) / 10 = 0.6

    // n = 6, k = 2, maxpts = 10
    // p_win1 1+1, 1+2, 1+3, 1+4, 1+5, 2,   3,  4,  5,  6
    //        0.01 0.01 0.01 0.01 0.01 0.1 0.1 0.1 0.1 0.1 = 0.55

    // n = 6, k = 3, maxpts = 10
    // p_win  1+1+1, 1+1+2, 1+1+3, 1+1+4, 1+2, 1+3, 1+4, 1+5, 2+1, 2+2, 2+3, 2+4, 3,  4,  5,   6
    //        0.001  0.001  0.001  0.001  0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.1 0.1 0.1 0.1
    // sum=0.484

    // n = 6, k = 4, maxpts = 10
    // p_win  1+1+1+1 1+1+1+2 1+1+1+3 1+1+2 1+1+3 1+1+4 1+2+1 1+2+2 1+2+3 1+3  1+4  1+5  2+1+1 2+1+2 2+1+3 2+2  2+3  2+4  3+1  3+2  3+3  4   5   6
    //         .0001   .0001   .0001   .001  .001  .001  .001  .001  .001  .01  .01  .01  .001  .001  .001  .01  .01  .01  .01  .01  .01  .1  .1  .1
    //sum=0.3993
    val p1 = 1.0 / maxPts.toDouble()
    val cache = Array<Double>(n + 1) { -1.0 }
        // f(x) = (f(x+1) + f(x+2)+..+f(x + maxPts))*p1
        // f(x + 1) = (f(x+2) + f(x+3) +...+f(x + maxPts)*p1 + f(x + 1 + maxPts))*p1
        // f(x) - f(x + 1) = f(x+1)*p1 - f(x+1+maxPts)*p1
        // f(x) = f(x+1) + (f(x+1) - f(x+1+maxPts))*p1
    fun dfs(currSum: Int): Double {
        if (currSum == k - 1) return minOf(1.0, (n-k+1)*p1) // corner case
        if (currSum >= k) return if (currSum <= n) 1.0 else 0.0
        if (cache[currSum] != -1.0) return cache[currSum]
        //var sumP = 0.0
        //for (x in 1..minOf(maxPts, n - currSum)) {
             //    sumP += dfs(currSum + x)
        //}
        //val res = sumP * p1
        val res = dfs(currSum + 1) + (dfs(currSum + 1) - dfs(currSum + 1 + maxPts)) * p1
        cache[currSum] = res
        return res
    }
    return dfs(0)
}

```

# 24.05.2023
[2542. Maximum Subsequence Score](https://leetcode.com/problems/maximum-subsequence-score/description/) medium
[blog post](https://leetcode.com/problems/maximum-subsequence-score/solutions/3557549/kotlin-priorityqueue/)
[substack](https://dmitriisamoilenko.substack.com/p/24052023-2542-maximum-subsequence?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/222
#### Problem TLDR
Max score of `k` sum(subsequence(a)) * min(subsequence(b))
#### Intuition
First, the result is independent of the order, so we can sort. For maximum score, it better to start with maximum multiplier of `min`. Then, we iterate from biggest nums2 to smallest. Greedily add numbers until we reach `k` elements. After `size > k`, we must consider what element to extract. Given our `min` is always the current value, we can safely take any element without modifying the minimum, thus take out the smallest by `nums1`.

#### Approach
* use `PriorityQueue` to dynamically take out the smallest
* careful to update score only when `size == k`, as it may decrease with more elements
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun maxScore(nums1: IntArray, nums2: IntArray, k: Int): Long {
    // 14  2 1 12 100000000000  1000000000000 100000000000
    // 13 11 7 1  1             1             1
    val inds = nums1.indices.sortedWith(
    compareByDescending<Int> { nums2[it] }
        .thenByDescending { nums1[it] })
    var score = 0L
    var sum = 0L
    val pq = PriorityQueue<Int>(compareBy { nums1[it] })
    inds.forEach {
        sum += nums1[it].toLong()
        pq.add(it)
        if (pq.size > k) sum -= nums1[pq.poll()].toLong()
        if (pq.size == k) score = maxOf(score, sum * nums2[it].toLong())
    }
    return score
}

```

# 23.05.2023
[703. Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/description/) medium
[blog post](https://leetcode.com/problems/kth-largest-element-in-a-stream/solutions/3554138/kotlin-priority-queue/)
[substack](https://dmitriisamoilenko.substack.com/p/23052023-703-kth-largest-element?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/221
#### Problem TLDR
Kth largest
#### Intuition
We need to keep all values smaller than current largest kth element and can safely drop all other elements.
#### Approach
Use `PriorityQueue`.
#### Complexity
- Time complexity:
$$O(nlogk)$$
- Space complexity:
$$O(k)$$

#### Code

```kotlin

class KthLargest(val k: Int, nums: IntArray) {
    val pq = PriorityQueue<Int>(nums.toList())

        fun add(v: Int): Int = with (pq) {
            add(v)
            while (size > k) poll()
            peek()
        }
    }

```

# 22.05.2023
[347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/) medium
[blog post](https://leetcode.com/problems/top-k-frequent-elements/solutions/3550637/kotlin-bucket-sort/)
[substack](https://dmitriisamoilenko.substack.com/p/22052023-347-top-k-frequent-elements?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/220
#### Problem TLDR
First `k` unique elements sorted by frequency.
#### Intuition
Group by frequency `1 1 1 5 5 -> 1:3, 5:2`, then bucket sort frequencies `2:5, 3:1`, then flatten and take first `k`.
#### Approach
* We can use [Kotlin collections api](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-map/)
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun topKFrequent(nums: IntArray, k: Int): IntArray {
    val freq = nums.groupBy { it }.mapValues { it.value.size }
    val freqToNum = Array<MutableList<Int>>(nums.size + 1) { mutableListOf() }
    freq.forEach { (num, fr) -> freqToNum[nums.size + 1 - fr].add(num) }
    return freqToNum
        .filter { it.isNotEmpty() }
        .flatten()
        .take(k)
        .toIntArray()
}

```

# 21.05.2023
[934. Shortest Bridge](https://leetcode.com/problems/shortest-bridge/description/) medium
[blog post](https://leetcode.com/problems/shortest-bridge/solutions/3546914/kotlin-dfs-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/21052023-934-shortest-bridge?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/219
#### Problem TLDR
Find the shortest path from one island of `1`'s to another.
#### Intuition
Shortest path can be found with Breadth-First Search if we start it from every border cell of the one of the islands.
To detect border cell, we have to make separate DFS.

#### Approach
* modify grid to store `visited` set
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun Array<IntArray>.inRange(xy: Pair<Int, Int>) = (0..lastIndex).let {
    xy.first in it && xy.second in it
}
fun Pair<Int, Int>.siblings() = arrayOf(
(first - 1) to second, first to (second - 1),
(first + 1) to second, first to (second + 1)
)
fun shortestBridge(grid: Array<IntArray>): Int {
    val queue = ArrayDeque<Pair<Int, Int>>()
    fun dfs(x: Int, y: Int) {
        if (grid[y][x] == 1) {
            grid[y][x] = 2
            (x to y).siblings().filter { grid.inRange(it) }.forEach { dfs(it.first, it.second) }
        } else if (grid[y][x] == 0) queue.add(x to y)
    }
    (0 until grid.size * grid.size)
    .map { it / grid.size to it % grid.size}
    .filter { (y, x) -> grid[y][x] == 1 }
    .first().let { (y, x) -> dfs(x, y)}
    return with (queue) {
        var steps = 1
        while (isNotEmpty()) {
            repeat(size) {
                val xy = poll()
                if (grid.inRange(xy)) {
                    val (x, y) = xy
                    if (grid[y][x] == 1) return@shortestBridge steps - 1
                    if (grid[y][x] == 0) {
                        grid[y][x] = 3
                        addAll(xy.siblings().filter { grid.inRange(it) })
                    }
                }
            }
            steps++
        }
        -1
    }
}

```

# 20.05.2023
[399. Evaluate Division](https://leetcode.com/problems/evaluate-division/description/) medium
[blog post](https://leetcode.com/problems/evaluate-division/solutions/3543427/kotlin-n-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/20052023-399-evaluate-division?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/218
#### Problem TLDR
Given values for `a/b` and `b/c` find answers for `a/c`.
#### Intuition
Let's build a graph, `a` -> `b` with weights of `values[a/b]`. Then answer is a path from one node to the other. The shortest path can be found with a Breadth-First Search.

#### Approach
* careful with corner case `x/x`, where `x` is not in a graph.
#### Complexity
- Time complexity:
$$O(nEV)$$
- Space complexity:
$$O(n+E+V)$$

#### Code

```kotlin

fun calcEquation(equations: List<List<String>>, values: DoubleArray, queries: List<List<String>>): DoubleArray {
    val fromTo = mutableMapOf<String, MutableList<Pair<String, Double>>>()
    equations.forEachIndexed { i, (from, to) ->
        fromTo.getOrPut(from) { mutableListOf() } += to to values[i]
        fromTo.getOrPut(to) { mutableListOf() } += from to (1.0 / values[i])
    }
    // a/c = a/b * b/c
    return queries.map { (from, to) ->
        with(ArrayDeque<Pair<String, Double>>()) {
            val visited = HashSet<String>()
                visited.add(from)
                if (fromTo.containsKey(to)) add(from to 1.0)
                while (isNotEmpty()) {
                    repeat(size) {
                        val (point, value) = poll()
                        if (point == to) return@map value
                        fromTo[point]?.forEach { (next, nvalue) ->
                            if (visited.add(next)) add(next to value * nvalue)
                        }
                    }
                }
                -1.0
            }
        }.toDoubleArray()
    }

```

# 19.05.2023
[785. Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/description/) medium
[blog post](https://leetcode.com/problems/is-graph-bipartite/solutions/3540319/kotlin-dfs-red-blue/)
[substack](https://dmitriisamoilenko.substack.com/p/19052023-785-is-graph-bipartite?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/217
#### Problem TLDR
Find if graph is [bipartite](https://en.wikipedia.org/wiki/Bipartite_graph)
#### Intuition
![image.png](https://assets.leetcode.com/users/images/7a6642c2-2c85-40c7-9e0e-9593c606c128_1684467332.1593482.png)
Mark edge `Red` or `Blue` and it's nodes in the opposite.

#### Approach
* there are disconnected nodes, so run DFS for all of them
#### Complexity
- Time complexity:
$$O(VE)$$, DFS once for all `vertices` and `edges`
- Space complexity:
$$O(V+E)$$, for `reds` and `visited` set.

#### Code

```kotlin

fun isBipartite(graph: Array<IntArray>): Boolean {
    val reds = IntArray(graph.size)
    fun dfs(u: Int, isRed: Int): Boolean {
        if (reds[u] == 0) {
            reds[u] = if (isRed == 0) 1 else isRed
            return graph[u].all { dfs(it, -reds[u]) }
        } else return reds[u] == isRed
    }
    return graph.indices.all { dfs(it, reds[it]) }
}

```

# 18.05.2023
[1557. Minimum Number of Vertices to Reach All Nodes](https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-vertices-to-reach-all-nodes/solutions/3536694/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/18052023-1557-minimum-number-of-vertices?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/216
#### Problem TLDR
Find all starting nodes in graph.
#### Intuition
Count nodes that have no incoming connections.

#### Approach
* we can use subtract operation in Kotlin
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun findSmallestSetOfVertices(n: Int, edges: List<List<Int>>): List<Int> =
    (0 until n) - edges.map { it[1] }

```

# 17.05.2023
[2130. Maximum Twin Sum of a Linked List](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/description/) medium
[blog post](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/solutions/3532758/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/17052023-2130-maximum-twin-sum-of?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/215
#### Problem TLDR
Max sum of head-tail twin ListNodes: `a-b-c-d -> max(a+d, b+c)`
#### Intuition
Add first half to the `Stack`, then pop until end reached.
#### Approach
* use `fast` and `slow` pointers to find the center.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code

```kotlin

        fun pairSum(head: ListNode?): Int {
            var fast = head
            var slow = head
            var sum = 0
            val stack = Stack<Int>()
                while (fast != null) {
                    stack.add(slow!!.`val`)
                    slow = slow.next
                    fast = fast.next?.next
                }
                while (slow != null) {
                    sum = maxOf(sum, stack.pop() + slow.`val`)
                    slow = slow.next
                }
                return sum
            }

```

# 16.05.2023
[24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/description/) medium
[blog post](https://leetcode.com/problems/swap-nodes-in-pairs/solutions/3529159/kotlin-be-explicit-to-avoid-bugs/)
[substack](https://dmitriisamoilenko.substack.com/p/16052023-24-swap-nodes-in-pairs?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/214
#### Problem TLDR
Swap adjacent ListNodes `a-b-c-d -> b-a-d-c`.
#### Intuition
Those kinds of problems are easy, but your task is to write it bug free from the first go.

#### Approach
For more robust code:
* use `dummy` head to track for a new head
* use explicit variables for each node in the configuration
* do debug code by writing down it values in the comments
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun swapPairs(head: ListNode?): ListNode? {
    val dummy = ListNode(0).apply { next = head }
    var curr: ListNode? = dummy
    while (curr?.next != null && curr?.next?.next != null) {
        // curr->one->two->next
        // curr->two->one->next
        var one = curr.next
        var two = one?.next
        val next = two?.next
        curr.next = two
        two?.next = one
        one?.next = next

        curr = one
    }
    return dummy.next
}

```

# 15.05.2023
[1721. Swapping Nodes in a Linked List](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/description/) medium
[blog post](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/solutions/3525455/kotlin-swap-values-not-nodes/)
[substack](https://dmitriisamoilenko.substack.com/p/15052023-1721-swapping-nodes-in-a?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/213
#### Problem TLDR
Swap the values of the head-tail k'th ListNodes.
#### Intuition
As we aren't asked to swap nodes, the problem is to find nodes.

#### Approach
Travel the `fast` pointer at `k` distance, then move both `fast` and `two` nodes until `fast` reaches the end.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun swapNodes(head: ListNode?, k: Int): ListNode? {
    var fast = head
    for (i in 1..k - 1) fast = fast?.next
    val one = fast
    var two = head
    while (fast?.next != null) {
        two = two?.next
        fast = fast?.next
    }
    one?.`val` = two?.`val`.also { two?.`val` = one?.`val` }
    return head
}

```

# 14.05.2023
[1799. Maximize Score After N Operations](https://leetcode.com/problems/maximize-score-after-n-operations/description/) hard
[blog post](https://leetcode.com/problems/maximize-score-after-n-operations/solutions/3522041/kotiln-dfs-cache-bitmask-gcd/)
[substack](https://dmitriisamoilenko.substack.com/p/14052023-1799-maximize-score-after?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/212
#### Problem TLDR
Max indexed-gcd-pair sum from 2n array; [3,4,6,8] -> 11 (1gcd(3,6) + 2gcd(4,8))
#### Intuition
For each `step` and remaining items, the result is always the same, so is memorizable.

#### Approach
* search all possible combinations with DFS
* use `bitmask` to avoid double counting
* use an array for cache
#### Complexity
- Time complexity:
$$O(n^22^n)$$
- Space complexity:
$$O(n2^n)$$

#### Code

```kotlin

    fun gcd(a: Int, b: Int): Int = if (b % a == 0) a else gcd(b % a, a)
    fun maxScore(nums: IntArray): Int {
        val n = nums.size / 2
        val cache = Array(n + 1) { IntArray(1 shl nums.size) { -1 } }
        fun dfs(step: Int, mask: Int): Int {
            if (step > n) return 0
            if (cache[step][mask] != -1) return cache[step][mask]
            var max = 0
            for (i in 0..nums.lastIndex) {
                val ibit = 1 shl i
                if (mask and ibit != 0) continue
                for (j in (i + 1)..nums.lastIndex) {
                    val jbit = 1 shl j
                    if (mask and jbit != 0) continue
                    val curr = step * gcd(nums[i], nums[j])
                    val next = dfs(step + 1, mask or ibit or jbit)
                    max = maxOf(max, curr + next)
                }
            }
            cache[step][mask] = max
            return max
        }
        return dfs(1, 0)
    }

```

# 13.05.2023
[2466. Count Ways To Build Good Strings](https://leetcode.com/problems/count-ways-to-build-good-strings/description/) medium
[blog post](https://leetcode.com/problems/count-ways-to-build-good-strings/solutions/3518102/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/13052023-2466-count-ways-to-build?sd=pf)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/211

#### Problem
Count distinct strings, length low to high, appending '0' zero or '1' one times. Return count % 1,000,000,007.

#### Intuition
Let's add `zero`'s or `one`'s one by one. For each current length, the resulting count is independent of all the previous additions. We can cache the result by the current `size` of the string.

#### Approach
Let's write a DFS solution, adding `zero` or `one` and count the good strings.
Then we can rewrite it to the iterative DP.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

#### Code
top-down:

```

fun countGoodStrings(low: Int, high: Int, zero: Int, one: Int): Int {
    val m = 1_000_000_007
    val cache = mutableMapOf<Int, Int>()
    fun dfs(currSize: Int): Int {
        if (currSize > high) return 0
        return cache.getOrPut(currSize) {
            val curr = if (currSize in low..high) 1 else 0
            val addZeros = if (zero > 0) dfs(currSize + zero) else 0
            val addOnes = if (one > 0) dfs(currSize + one) else 0
            (curr + addZeros + addOnes) % m
        }
    }
    return dfs(0)
}

```

bottom-up

```

fun countGoodStrings(low: Int, high: Int, zero: Int, one: Int): Int {
    val cache = mutableMapOf<Int, Int>()
    for (sz in high downTo 0)
    cache[sz] = ((if (sz >= low) 1 else 0)
    + (cache[sz + zero]?:0)
    + (cache[sz + one]?:0)) % 1_000_000_007
    return cache[0]!!
}

```

# 12.05.2023
[2140. Solving Questions With Brainpower](https://leetcode.com/problems/solving-questions-with-brainpower/description/) medium

```kotlin

fun mostPoints(questions: Array<IntArray>): Long {
    val dp = LongArray(questions.size)
    for (i in questions.lastIndex downTo 0) {
        val (points, skip) = questions[i]
        val tail = if (i + skip + 1 > questions.lastIndex) 0 else dp[i + skip + 1]
        val notTake = if (i + 1 > questions.lastIndex) 0 else dp[i + 1]
        dp[i] = maxOf(points + tail, notTake)
    }
    return dp[0]
}

```

or minified golf version

```

fun mostPoints(questions: Array<IntArray>): Long {
    val dp = HashMap<Int, Long>()
    for ((i, q) in questions.withIndex().reversed())
    dp[i] = maxOf(q[0] + (dp[i + q[1] + 1]?:0), dp[i + 1]?:0)
    return dp[0]?:0
}

```

[blog post](https://leetcode.com/problems/solving-questions-with-brainpower/solutions/3514521/kotlin-dp/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-12052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/210
#### Intuition
If we go from the tail, for each element we are interested only on what happens to the `right` from it. Prefix of the array is irrelevant, when we're starting from the element `i`, because we sure know, that we are taking it and not skipping.
Given that, dynamic programming equation is:
$$dp_i = max(points_i + dp_{i+1+skip_i}, dp_{i+1})$$, where `dp` is the `mostPoints` starting from position `i`.

#### Approach
Let's implement a bottom-up solution.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 11.05.2023
[1035. Uncrossed Lines](https://leetcode.com/problems/uncrossed-lines/description/) medium

```kotlin

fun maxUncrossedLines(nums1: IntArray, nums2: IntArray): Int {
    val cache = Array(nums1.size) { Array(nums2.size) { -1 } }
    val intersect = nums1.toSet().intersect(nums2.toSet())

    fun dfs(i: Int, j: Int, x: Int): Int {
        if (i == nums1.size || j == nums2.size) return 0
        val cached = cache[i][j]
        if (cached != -1) return cached
        val n1 = nums1[i]
        val n2 = nums2[j]
        val drawLine = if (n1 == x && n2 == x || n1 == n2) 1 + dfs(i + 1, j + 1, n1) else 0
        val skipTop = dfs(i + 1, j, x)
        val skipBottom = dfs(i, j + 1, x)
        val skipBoth = dfs(i + 1, j + 1, x)
        val startTop = if (intersect.contains(n1)) dfs(i + 1, j, n1) else 0
        val startBottom = if (intersect.contains(n2)) dfs(i, j + 1, n2) else 0
        val res = maxOf(
        drawLine,
        maxOf(drawLine, skipTop, skipBottom),
        maxOf(skipBoth, startTop, startBottom)
        )
        cache[i][j] = res
        return res
    }
    return dfs(0, 0, 0)
}

```

[blog post](https://leetcode.com/problems/uncrossed-lines/solutions/3510891/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-11052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/209
#### Intuition
Consider the case:

```

2 5 1 2 5
2 2 2 1 1 1 5 5 5

```

![image.png](https://assets.leetcode.com/users/images/61b85278-34db-4858-b235-610bf4553518_1683778465.8258896.png)

When we draw all the possible lines, we see that there is a choice to draw line `2-2` or four lines `1-1` or three `5-5` in the middle. Suffix lines `5-5` and prefix lines `2-2` are optimal already and can be cached as a result.
To find an optimal choice we can use DFS.
We can prune some impossible combinations by precomputing the intersected numbers and considering them only.
#### Approach
* use an array for the faster cache instead of HashMap
* for the intersection there is an `intersect` method in Kotlin

#### Complexity
- Time complexity:
$$O(n^3)$$
- Space complexity:
$$O(n^3)$$

# 10.05.2023
[59. Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/description/) medium

```kotlin

fun generateMatrix(n: Int): Array<IntArray> = Array(n) { IntArray(n) }.apply {
    var dir = 0
    var dxdy = arrayOf(0, 1, 0, -1)
    var x = 0
    var y = 0
    val nextX = { x + dxdy[(dir + 1) % 4] }
    val nextY = { y + dxdy[dir] }
    val valid = { x: Int, y: Int -> x in 0..n-1 && y in 0..n-1 && this[y][x] == 0 }

    repeat (n * n) {
        this[y][x] = it + 1
        if (!valid(nextX(), nextY())) dir = (dir + 1) % 4
        x = nextX()
        y = nextY()
    }
}

```

[blog post](https://leetcode.com/problems/spiral-matrix-ii/solutions/3506921/kotlin-a-robot/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-10052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/208
#### Intuition
Just implement what is asked. Let's have the strategy of a robot: move it in one direction until it hits a wall, then change the direction.

#### Approach
* to detect an empty cell, we can check it for `== 0`
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

# 9.05.2023
[54. Spiral Matrix](https://leetcode.com/problems/spiral-matrix/description/) medium

```kotlin

fun spiralOrder(matrix: Array<IntArray>): List<Int> = mutableListOf<Int>().apply {
    var x = 0
    var y = 0
    val dxy = arrayOf(0, 1, 0, -1)
    val borders = arrayOf(matrix[0].lastIndex, matrix.lastIndex, 0, 0)
    var dir = 0
    val moveBorder = { border: Int -> borders[border] += if (border < 2) -1 else 1 }
    repeat (matrix.size * matrix[0].size) {
        if ((if (dir % 2 == 0) x else y) == borders[dir]) {
            moveBorder((dir + 3) % 4)
            dir = (dir + 1) % 4
        }
        add(matrix[y][x])
        x += dxy[(dir + 1) % 4]
        y += dxy[dir]
    }
}

```

[blog post](https://leetcode.com/problems/spiral-matrix/solutions/3503485/kotlin-robot/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-9052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/207
#### Intuition
Just implement what is asked.
We can use a loop with four directions, or try to program `a robot` that will rotate after it hit a wall.

#### Approach
* do track the borders `left`, `top`, `right`, `bottom`
* use single direction variable `dir`
* move the wall after a robot walked parallel to it
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 8.05.2023
[1572. Matrix Diagonal Sum](https://leetcode.com/problems/matrix-diagonal-sum/description/) easy

```kotlin

fun diagonalSum(mat: Array<IntArray>): Int =
    (0..mat.lastIndex).sumBy {
        mat[it][it] + mat[it][mat.lastIndex - it]
    }!! - if (mat.size % 2 == 0) 0 else mat[mat.size / 2][mat.size / 2]

```

[blog post](https://leetcode.com/problems/matrix-diagonal-sum/solutions/3498716/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-8052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/206
#### Intuition
Just do what is asked.
#### Approach
* avoid double counting of the center element
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 7.05.2023
[1964. Find the Longest Valid Obstacle Course at Each Position](https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/description/) hard

```kotlin

fun longestObstacleCourseAtEachPosition(obstacles: IntArray): IntArray {
    // 2 3 1 3
    // 2          2
    //   3        2 3
    //     1      1 3    (pos = 1)
    //       3    1 3 3

    // 5 2 5 4 1 1 1 5 3 1
    // 5       .             5
    //   2     .             2
    //     5   .             2 5
    //       4 .             2 4
    //         1             1 4 (pos = 1)
    //           1           1 1
    //             1         1 1 1
    //               5       1 1 1 5
    //                 3     1 1 1 3
    //                   1   1 1 1 1

    val lis = IntArray(obstacles.size)
    var end = 0
    return obstacles.map { x ->
        var pos = -1
        var lo = 0
        var hi = end - 1
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (lis[mid] > x) {
                hi = mid - 1
                pos = mid
            } else lo = mid + 1
        }
        if (pos == -1) {
            lis[end++] = x
            end
        } else {
            lis[pos] = x
            pos + 1
        }
    }.toIntArray()
}

```

[blog post](https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/solutions/3495432/kotlin-lis/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-7052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/205
#### Intuition
This is the [Longest Increasing Subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence) length problem, that have a classic algorithm, which must be learned and understood.

The trivial case of `any increasing subsequence` is broken by example: `2 3 1 3`, when we consider the last `3` result must be: `233` instead of `13`. So, we must track all the sequences.

To track all the sequences, we can use `TreeMap` that will hold the `largest` element and length of any subsequence. Adding a new element will take $$O(n^2)$$.

The optimal `LIS` solution is to keep the largest increasing subsequence so far and cleverly add new elements:
1. for a new element, search for the `smallest` element that is `larger` than it
2. if found, replace
3. if not, append
![lis.gif](https://assets.leetcode.com/users/images/0d26a398-07fa-4653-acab-3e02564051d4_1683437400.728855.gif)

#### Approach
* google what is the solution of `LIS`
* use an array for `lis`
* carefully write binary search
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

# 6.05.2023
[1498. Number of Subsequences That Satisfy the Given Sum Condition](https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/description/) medium

```kotlin

fun numSubseq(nums: IntArray, target: Int): Int {
    val m = 1_000_000_007
    nums.sort()
    val cache = IntArray(nums.size + 1) { 0 }
    cache[1] = 1
    for (i in 2..nums.size) cache[i] = (2 * cache[i - 1]) % m
    var total = 0
    nums.forEachIndexed { i, n ->
        var lo = 0
        var hi = i
        var removed = cache[i + 1]
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (nums[mid] + n <= target) {
                removed = cache[i - mid]
                lo = mid + 1
            } else hi = mid - 1
        }
        total = (total + cache[i + 1] - removed) % m
    }
    if (total < 0) total += m
    return total
}

```

[blog post](https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/solutions/3492072/kotlin-this-problem-is-hard/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-6052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/204
#### Intuition
1. We can safely sort an array, because order doesn't matter for finding `max` or `min` in a subsequence.
2. Having increasing order gives us the pattern:
![image.png](https://assets.leetcode.com/users/images/a8c6d686-b4ca-40bb-98ab-9579d6a9df26_1683355137.4598198.png)
Ignoring the `target`, each new number adds previous value to the sum: $$sum_2 = sum_1 + (1 + sum_1)$$, or just $$2^i$$.
3. Let's observe the pattern of the removed items:
![image.png](https://assets.leetcode.com/users/images/06090902-7c9e-4df1-8880-b7f238ae7e17_1683355450.981601.png)
For example, `target = 12`, for number `8`, count of excluded values is `4` = [568, 58, 68, 8]; for number `9`, it is `8` = [5689, 589, 569, 59, 689, 69, 89, 9]. We can observe, it is determined by the sequence `5 6 8 9`, where all the numbers are bigger, than `target - 9`. That is, the law for excluding the elements is the same: $$r_2 = r_1 + (1 + r_1)$$, or just $$2^x$$, where x - is the count of the bigger numbers.

#### Approach
* Precompute the 2-powers
* Use binary search to count how many numbers are out of the equation `n_i + x <= target`
* A negative result can be converted to positive by adding the modulo `1_000_000_7`
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

# 5.05.2023
[1456. Maximum Number of Vowels in a Substring of Given Length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/) medium

```kotlin

fun maxVowels(s: String, k: Int): Int {
    val vowels = setOf('a', 'e', 'i', 'o', 'u')
    var count = 0
    var max = 0
    for (i in 0..s.lastIndex) {
        if (s[i] in vowels) count++
        if (i >= k && s[i - k] in vowels) count--
        if (count > max) max = count
    }
    return max
}

```

[blog post](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/solutions/3487078/kotlin-sliding-window/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-5052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/203
#### Intuition
Count vowels, increasing them on the right border and decreasing on the left of the sliding window.
#### Approach
* we can use `Set` to check if it is a vowel
* look at `a[i - k]` to detect if we must start move left border from `i == k`
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 4.05.2023
[649. Dota2 Senate](https://leetcode.com/problems/dota2-senate/description/) medium

```kotlin

fun predictPartyVictory(senate: String): String {
    val queue = ArrayDeque<Char>()
        senate.forEach { queue.add(it) }
        var banR = 0
        var banD = 0
        while (true) {
            var haveR = false
            var haveD = false
            repeat(queue.size) {
                val c = queue.poll()
                if (c == 'R') {
                    haveR = true
                    if (banR > 0) banR--
                    else {
                        queue.add(c)
                        banD++
                    }
                } else {
                    haveD = true
                    if (banD > 0) banD--
                    else {
                        queue.add(c)
                        banR++
                    }
                }
            }
            if (!haveR) return "Dire"
            if (!haveD) return "Radiant"
        }
    }

```

[blog post](https://leetcode.com/problems/dota2-senate/solutions/3483710/kotlin-simulation/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-4052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/202
#### Intuition
*One can ban on any length to the right.*
We can just simulate the process, and it will take at most two rounds.

#### Approach
Use `Queue` and count how many bans are from the Radiant and from the Dire.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 3.05.2023
[2215. Find the Difference of Two Arrays](https://leetcode.com/problems/find-the-difference-of-two-arrays/description/) easy

```kotlin

fun findDifference(nums1: IntArray, nums2: IntArray): List<List<Int>> = listOf(
    nums1.subtract(nums2.toSet()).toList(),
    nums2.subtract(nums1.toSet()).toList()
    )

```

[blog post](https://leetcode.com/problems/find-the-difference-of-two-arrays/solutions/3479943/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-3052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/201
#### Intuition
Just do what is asked.

#### Approach
One way is to use two `Sets` and just filter them.
Another is to use `intersect` and `distinct`.
Third option is to sort both of them and iterate, that will use $$O(1)$$ extra memory, but $$O(nlogn)$$ time.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 2.05.2023
[1822. Sign of the Product of an Array](https://leetcode.com/problems/sign-of-the-product-of-an-array/description/) easy

```kotlin

fun arraySign(nums: IntArray): Int = nums.fold(1) { r, t -> if (t == 0) 0 else r * (t / Math.abs(t)) }

```

[blog post](https://leetcode.com/problems/sign-of-the-product-of-an-array/solutions/3475973/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-2052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/199
#### Intuition
Do what is asked, but avoid overflow.

#### Approach
There is an `sign` function in kotlin, but leetcode.com doesn't support it yet.
We can use `fold`.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 1.05.2023
[1491. Average Salary Excluding the Minimum and Maximum Salary](https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/description/) easy

```kotlin

fun average(salary: IntArray): Double = with (salary) {
    (sum() - max()!! - min()!!) / (size - 2).toDouble()
}

```

or

```

fun average(salary: IntArray): Double = salary.sorted().drop(1).dropLast(1).average()

```

[blog post](https://leetcode.com/problems/average-salary-excluding-the-minimum-and-maximum-salary/solutions/3471763/kotlin-sum-max-min/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-1052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/198
#### Intuition
Just do what is asked.

#### Approach
We can do `.fold` and iterate only once, but `sum`, `max` and `min` operators are less verbose.
We also can sort it, that will make code even shorter.
#### Complexity
- Time complexity:
$$O(n)$$, $$O(nlog(n))$$ for sorted
- Space complexity:
$$O(1)$$

# 30.04.2023
[1579. Remove Max Number of Edges to Keep Graph Fully Traversable](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/description/) hard

```kotlin

fun IntArray.root(a: Int): Int {
    var x = a
    while (this[x] != x) x = this[x]
    this[a] = x
    return x
}
fun IntArray.union(a: Int, b: Int): Boolean {
    val rootA = root(a)
    val rootB = root(b)
    if (rootA != rootB) this[rootB] = rootA
    return rootA != rootB
}
fun IntArray.connected(a: Int, b: Int) = root(a) == root(b)
fun maxNumEdgesToRemove(n: Int, edges: Array<IntArray>): Int {
    val uf1 = IntArray(n + 1) { it }
    val uf2 = IntArray(n + 1) { it }
    var skipped = 0
    edges.forEach { (type, a, b) ->
        if (type == 3) {
            uf1.union(a, b)
            if (!uf2.union(a, b)) skipped++
        }
    }
    edges.forEach { (type, a, b) ->
        if (type == 1 && !uf1.union(a, b)) skipped++
    }
    edges.forEach { (type, a, b) ->
        if (type == 2 && !uf2.union(a, b)) skipped++
    }
    for (i in 2..n)
    if (!uf1.connected(i - 1, i) || !uf2.connected(i - 1, i)) return -1
    return skipped
}

```

[blog post](https://leetcode.com/problems/remove-max-number-of-edges-to-keep-graph-fully-traversable/solutions/3468491/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-30042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/196
#### Intuition
After connecting all `type 3` nodes, we can skip already connected nodes for Alice and for Bob. To detect if all the nodes are connected, we can just check if all nodes connected to one particular node.
#### Approach
Use separate `Union-Find` objects for Alice and for Bob
#### Complexity
- Time complexity:
$$O(n)$$, as `root` and `union` operations take `< 5` for any `n <= Int.MAX`.
- Space complexity:
$$O(n)$$

# 29.04.2023
[1697. Checking Existence of Edge Length Limited Paths](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/description/) hard

```kotlin

fun distanceLimitedPathsExist(n: Int, edgeList: Array<IntArray>, queries: Array<IntArray>): BooleanArray {
    val uf = IntArray(n) { it }
    fun root(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun union(a: Int, b: Int) {
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) uf[rootB] = rootA
    }
    val indices = queries.indices.sortedWith(compareBy( { queries[it][2] } ))
    edgeList.sortWith(compareBy( { it[2] } ))
    var edgePos = 0
    val res = BooleanArray(queries.size)
    indices.forEach { ind ->
        val (qfrom, qto, maxDist) = queries[ind]
        while (edgePos < edgeList.size) {
            val (from, to, dist) = edgeList[edgePos]
            if (dist >= maxDist) break
            union(from, to)
            edgePos++
        }
        res[ind] = root(qfrom) == root(qto)
    }
    return res
}

```

[blog post](https://leetcode.com/problems/checking-existence-of-edge-length-limited-paths/solutions/3465266/kotlin-union-islands/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-29042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/195
#### Intuition
The naive approach is to do BFS for each query, obviously gives TLE as it takes $$O(n^2)$$ time.
Using the hint, we can use somehow the sorted order of the queries. If we connect every two nodes with `dist < query.dist` we have connected groups with all nodes reachable inside them. The best data structure for union and finding connected groups is the Union-Find.
To avoid iterating `edgeList` every time, we can sort it too and take only available distances.

#### Approach
* for better time complexity, compress the Union-Find path `uf[x] = n`
* track the `edgePos` - a position in a sorted `edgeList`
* make separate `indices` list to sort queries without losing the order
#### Complexity
- Time complexity:
$$O(nlog(n))$$, time complexity for `root` and `union` operations is an inverse Ackerman function and `< 5` for every possible number in Int.
- Space complexity:
$$O(n)$$

# 28.04.2023
[839. Similar String Groups](https://leetcode.com/problems/similar-string-groups/description/) hard

```kotlin

fun numSimilarGroups(strs: Array<String>): Int {
    fun similar(i: Int, j: Int): Boolean {
        var from = 0
        while (from < strs[i].length && strs[i][from] == strs[j][from]) from++
        var to = strs[i].lastIndex
        while (to >= 0 && strs[i][to] == strs[j][to]) to--
        for (x in from + 1..to - 1)
        if (strs[i][x] != strs[j][x]) return false
        return true
    }
    val uf = IntArray(strs.size) { it }
    fun root(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    var groups = strs.size
    fun union(a: Int, b: Int) {
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) {
            groups--
            uf[rootB] = rootA
        }
    }
    for (i in 0..strs.lastIndex)
    for (j in i + 1..strs.lastIndex)
    if (similar(i, j)) union(i, j)
    return groups
}

```

[blog post](https://leetcode.com/problems/similar-string-groups/solutions/3462309/kotlin-union-find/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-28042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/194
#### Intuition
For tracking the groups, Union-Find is a good start. Next, we need to compare the similarity of each to each word, that is $$O(n^2)$$.
For the similarity, we need a linear algorithm. Let's divide the words into three parts: `prefix+a+body+b+suffix`. Two words are similar if their `prefix`, `suffix` and `body` are similar, leaving the only different letters `a` and `b`.

#### Approach
* decrease the groups when the two groups are joined together
* shorten the Union-Find root's path `uf[x] = n`
* more complex Union-Find algorithm with `ranks` give the optimal time of $$O(lg*n)$$, where `lg*n` is the inverse Ackerman function. It is inverse of the f(n) = 2^2^2^2..n times.
#### Complexity
- Time complexity:
$$O(n^2a(n))$$
- Space complexity:
$$O(n)$$

# 27.04.2023
[319. Bulb Switcher](https://leetcode.com/problems/bulb-switcher/description/) medium

```kotlin

fun bulbSwitch(n: Int): Int {
    if (n <= 1) return n
    var count = 1
    var interval = 3
    var x = 1
    while (x + interval <= n) {
        x = x + interval
        interval += 2
        count++
    }
    return count
}

```

[blog post](https://leetcode.com/problems/bulb-switcher/solutions/3459491/kotlin-spot-the-pattern/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/leetcode-daily-27042023)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/193
#### Intuition
Let's draw a diagram and see if any pattern here:

```

//      1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
//
// 1    1 1 1 1 1 1 1 1 1  1 1  1  1  1  1  1  1  1  1
// 2      0 . 0 . 0 . 0 .  0 .  0  .  0  .  0  .  0  .
// 3        0 . . 1 . . 0  . .  1  .  .  0  .  .  1  .
// 4          1 . . . 1 .  . .  0  .  .  .  1  .  .  .
// 5            0 . . . .  1 .  .  .  .  1  .  .  .  .
// 6              0 . . .  . .  1  .  .  .  .  .  0  .
// 7                0 . .  . .  .  .  1  .  .  .  .  .
// 8                  0 .  . .  .  .  .  .  0  .  .  .
// 9                    1  . .  .  .  .  .  .  .  1  .
// 10                      0 .  .  .  .  .  .  .  .  .
// 11                        0  .  .  .  .  .  .  .  .
// 12                           0  .  .  .  .  .  .  .
// 13                              0  .  .  .  .  .  .
// 14                                 0  .  .  .  .  .
// 15                                    0  .  .  .  .
// 16                                       1  .  .  .
// 17                                          0  .  .
// 18                                             0  .
// 19                                                0

```

One rule is: number of switches for each new value is a number of divisors.
Another rule: we can reuse the previous result.
However, those rules didn't help much, let's observe another pattern: `diagonal sequence have increasing intervals of zeros by 2`

#### Approach
Use found law and write the code.
#### Complexity
- Time complexity:
That is tricky, let's derive it:
$$
n = 1 + 2 + (1+2+2) + (1+2+2+2) + (...) + (1+2k)
$$, or
$$
n = \sum_{i=0}^{k}1+2i = k(1 + 2 + 1 + 2k)/2
$$, then count of elements in arithmetic progression `k` is:
$$
O(k) = O(\sqrt{n})
$$, which is our time complexity.
- Space complexity:
$$O(1)$$

# 26.04.2023
[258. Add Digits](https://leetcode.com/problems/add-digits/description/) easy

```kotlin

fun addDigits(num: Int): Int = if (num == 0) 0 else 1 + ((num - 1) % 9)
// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38
// 0 1 2 3 4 5 6 7 8 9 1  2  3  4  5  6  7  8  9  1  2  3  4  5  6  7  8  9  1  2  3  4  5  6  7  8  9  1  2
// 0 [1..9] [1..9] [1..9] ...

```

[blog post](https://leetcode.com/problems/add-digits/solutions/3455825/kotlin-pattern/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-26042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/192
#### Intuition
Observing the pattern:

```

// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38
// 0 1 2 3 4 5 6 7 8 9 1  2  3  4  5  6  7  8  9  1  2  3  4  5  6  7  8  9  1  2  3  4  5  6  7  8  9  1  2
// 0 [1..9] [1..9] [1..9] ...

```

There is a repeating part of it: `[1..9]`, so we can derive the formula.

#### Approach
It is just an array pointer loop shifted by 1.
#### Complexity
- Time complexity:
$$O(1)$$
- Space complexity:
$$O(1)$$

# 25.04.2023
[2336. Smallest Number in Infinite Set](https://leetcode.com/problems/smallest-number-in-infinite-set/description/) medium

```kotlin

class SmallestInfiniteSet() {
    val links = IntArray(1001) { it + 1 }

    fun popSmallest(): Int {
        val smallest = links[0]
        val next = links[smallest]
        links[smallest] = 0
        links[0] = next
        return smallest
    }

    fun addBack(num: Int) {
        if (links[num] == 0) {
            var maxLink = 0
            while (links[maxLink] <= num) maxLink = links[maxLink]
            val next = links[maxLink]
            links[maxLink] = num
            links[num] = next
        }
    }

}

```

[blog post](https://leetcode.com/problems/smallest-number-in-infinite-set/solutions/3452738/kotlin-sparse-array/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-25042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/191
#### Intuition
Given the constraints, we can hold every element as a link node to another in an Array. This will give us $$O(1)$$ time for `pop` operation, but $$O(n)$$ for `addBack` in the worst case.
A more asymptotically optimal solution, is to use a `TreeSet` and a single pointer to the largest popped element.

#### Approach
Let's implement a sparse array.
##### Complexity
- Time complexity:
$$O(1)$$ - for `pop`
$$O(n)$$ - constructor and `addBack`
- Space complexity:
$$O(n)$$

# 24.04.2023
[1046. Last Stone Weight](https://leetcode.com/problems/last-stone-weight/description/) easy

```kotlin

fun lastStoneWeight(stones: IntArray): Int =
with(PriorityQueue<Int>(compareByDescending { it } )) {
    stones.forEach { add(it) }
    while (size > 1) add(poll() - poll())
    if (isEmpty()) 0 else peek()
}

```

[blog post](https://leetcode.com/problems/last-stone-weight/solutions/3449145/kotlin-priority-queue/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-24042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/190
#### Intuition
Just run the simulation.

#### Approach
* use `PriorityQueue` with `compareByDescending`
#### Complexity
- Time complexity:
$$O(nlog(n))$$
- Space complexity:
$$O(n)$$

# 23.04.2023
[1416. Restore The Array](https://leetcode.com/problems/restore-the-array/description/) hard

```kotlin

fun numberOfArrays(s: String, k: Int): Int {
    // 131,7  k=1000
    // 1317 > 1000
    // 20001  k=2000
    // 2      count=1
    //  000   count=1, curr=2000
    //     1  count++, curr=1
    //
    // 220001 k=2000
    // 2      count=1 curr=1
    // 22     count+1=2 curr=22          [2, 2], [22]
    // 220    curr=220                   [2, 20], [220]
    // 2200   curr=2200 > 2000, curr=200 [2, 200], [2200]
    // 22000  curr=2000   count=1        [2, 2000]
    // 220001 count+1=3 curr=20001 > 2000, curr=1  [2, 2000, 1], []
    val m = 1_000_000_007L
    val cache = LongArray(s.length) { -1L }
    fun dfs(curr: Int): Long {
        if (curr == s.length) return 1L
        if (s[curr] == '0') return 0L
        if (cache[curr] != -1L) return cache[curr]
        var count = 0L
        var num = 0L
        for (i in curr..s.lastIndex) {
            val d = s[i].toLong() - '0'.toLong()
            num = num * 10L + d
            if (num > k) break
            val countOther = dfs(i + 1)
            count = (count + countOther) % m
        }
        cache[curr] = count
        return count
    }
    return dfs(0).toInt()
}

or bottom-up

fun numberOfArrays(s: String, k: Int): Int {
    val cache = LongArray(s.length)
    for (curr in s.lastIndex downTo 0) {
        if (s[curr] == '0') continue
        var count = 0L
        var num = 0L
        for (i in curr..s.lastIndex) {
            num = num * 10L + s[i].toLong() - '0'.toLong()
            if (num > k) break
            val next = if (i == s.lastIndex) 1 else cache[i + 1]
            count = (count + next) % 1_000_000_007L
        }
        cache[curr] = count
    }
    return cache[0].toInt()
}

memory optimization:

fun numberOfArrays(s: String, k: Int): Int {
    val cache = LongArray(k.toString().length + 1)
    for (curr in s.lastIndex downTo 0) {
        System.arraycopy(cache, 0, cache, 1, cache.size - 1)
        if (s[curr] == '0') {
            cache[0] = 0
            continue
        }

        var count = 0L
        var num = 0L
        for (i in curr..s.lastIndex) {
            num = num * 10L + s[i].toLong() - '0'.toLong()
            if (num > k) break
            val next = if (i == s.lastIndex) 1 else cache[i - curr + 1]
            count = (count + next) % 1_000_000_007L
        }
        cache[0] = count
    }
    return cache[0].toInt()
}

```

[blog post](https://leetcode.com/problems/restore-the-array/solutions/3446057/kotlin-choose-dp-rule/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-23042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/189
#### Intuition
One naive solution, is to find all the possible ways of splitting the string, and calculating `soFar` number, gives TLE as we must take `soFar` into consideration when memoizing the result.
Let's consider, that for every position in `s` there is only one number of possible arrays. Given that, we can start from each position and try to take the `first` number in all possible correct ways, such that `num < k`. Now, we can cache this result for reuse.

#### Approach
* use `Long` to avoid overflow
* we actually not need all the numbers in cache, just the $$lg(k)$$ for the max length of the number
#### Complexity
- Time complexity:
$$O(nlg(k))$$
- Space complexity:
$$O(lg(k))$$

# 22.04.2023
[1312. Minimum Insertion Steps to Make a String Palindrome](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/description/) hard

```kotlin

fun minInsertions(s: String): Int {
    // abb -> abba
    // abb*
    //  ab -> aba / bab
    // *ab
    //  bba -> abba
    // *bba
    //   bbbaa -> aabbbaa
    // **bbbaa
    //    bbbcaa -> aacbbbcaa
    // ***bbbcaa
    // leetcode ->  leetcodocteel
    // leetcod***e**
    //      o -> 0
    //      od -> dod / dod -> 1
    //     cod -> codoc / docod -> 2
    //     code -> codedoc / edocode -> 2+1=3
    //    tcod -> tcodoct / doctcod -> 2+1=3
    //    tcode -> tcodedoct / edoctcode -> 3+1=4
    //   etcode = e{tcod}e -> e{tcodoct / doctcod}e -> 3
    //   etcod -> 1+{tcod} -> 1+3=4
    //  eetcod -> docteetcod 4 ?/ eetcodoctee 5
    //  eetcode -> edocteetcode 5 / eetcodedoctee 6 -> e{etcod}e 4 = e{etcodocte}e
    // leetcod -> 1+{eetcod} -> 5
    // leetcode -> 1+{eetcode} 1+4=5
    // aboba
    // a -> 0
    // ab -> 1
    // abo -> min({ab}+1, 1+{bo}) =2
    // abob -> min(1+{bob}, {abo} +1)=1
    // aboba -> min(0 + {bob}, 1+{abob}, 1+{boba}) = 0
    val cache = mutableMapOf<Pair<Int, Int>, Int>()
    fun dfs(from: Int, to: Int): Int {
        if (from > to || from < 0 || to > s.lastIndex) return -1
        if (from == to) return 0
        if (from + 1 == to) return if (s[from] == s[to]) 0 else 1
        return cache.getOrPut(from to to) {
            if (s[from] == s[to]) return@getOrPut dfs(from + 1, to - 1)
            val one = dfs(from + 1, to)
            val two = dfs(from, to - 1)
            when {
                one != -1 && two != -1 -> 1 + minOf(one, two)
                one != -1 -> 1 + one
                two != -1 -> 1 + two
                else -> -1
            }
        }
    }
    return dfs(0, s.lastIndex)
}

```

[blog post](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/solutions/3442679/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-22042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/188
#### Intuition
Let's add chars one by one. Single char is a palindrome. Two chars are palindrome if they are equal, and not if not: $$
insertions_{ab} =\begin{cases}
0, & \text{if a==b}\\
1
\end{cases}
$$. While adding a new character, we choose the minimum insertions. For example, `aboba`:

```

// aboba
// a -> 0
// ab -> 1
// abo -> min({ab}+1, 1+{bo}) =2
// abob -> min(1+{bob}, {abo} +1)=1
// aboba -> min(0 + {bob}, 1+{abob}, 1+{boba}) = 0

```

So, the DP equation is the following $$dp_{i,j} = min(0 + dp_{i+1, j-1}, 1 + dp_{i+1, j}, 1 + dp_{i, j-1}$$, where DP - is the minimum number of insertions.
#### Approach
Just DFS and cache.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

# 21.04.2023
[879. Profitable Schemes](https://leetcode.com/problems/profitable-schemes/description/) hard

```kotlin

fun profitableSchemes(n: Int, minProfit: Int, group: IntArray, profit: IntArray): Int {
    val cache = Array(group.size) { Array(n + 1) { Array(minProfit + 1) { -1 } } }
    fun dfs(curr: Int, guys: Int, cashIn: Int): Int {
        if (guys < 0) return 0
        val cash = minOf(cashIn, minProfit)
        if (curr == group.size) return if (cash == minProfit) 1 else 0
        with(cache[curr][guys][cash]) { if (this != -1) return@dfs this }
        val notTake = dfs(curr + 1, guys, cash)
        val take = dfs(curr + 1, guys - group[curr], cash + profit[curr])
        val res = (notTake + take) % 1_000_000_007
        cache[curr][guys][cash] = res
        return res
    }
    return dfs(0, n, 0)
}

```

[blog post](https://leetcode.com/problems/profitable-schemes/solutions/3439827/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-21042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/187
#### Intuition
For every new item, `j` we can decide to take it or not take it. Given the inputs of how many `guys` we have and how much `cash` already earned, the result is always the same: $$count_j = notTake_j(cash, guys) + take_j(cash + profit[j], guys - group[j])$$

#### Approach
Do DFS and cache result in an array.
#### Complexity
- Time complexity:
$$O(n^3)$$
- Space complexity:
$$O(n^3)$$

# 20.04.2023
[662. Maximum Width of Binary Tree](https://leetcode.com/problems/maximum-width-of-binary-tree/description/) medium

```kotlin

fun widthOfBinaryTree(root: TreeNode?): Int =
with(ArrayDeque<Pair<Int, TreeNode>>()) {
    root?.let { add(0 to it) }
    var width = 0
    while (isNotEmpty()) {
        var first = peek()
        var last = last()
        width = maxOf(width, last.first - first.first + 1)
        repeat(size) {
            val (x, node) = poll()
            node.left?.let { add(2 * x + 1 to it) }
            node.right?.let { add(2 * x + 2 to it) }
        }
    }
    width
}

```

[blog post](https://leetcode.com/problems/maximum-width-of-binary-tree/solutions/3436856/kotlin-bfs/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-20042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/186
#### Intuition
For every node, positions of it's left child is $$2x +1$$ and right is $$2x + 2$$
![leetcode_tree.gif](https://assets.leetcode.com/users/images/d1333748-b007-4b6d-85a3-71f073644b70_1681965037.4414012.gif)

#### Approach
We can do BFS and track node positions.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 19.04.2023
[1372. Longest ZigZag Path in a Binary Tree](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/description/) medium

```kotlin

fun longestZigZag(root: TreeNode?): Int {
    var max = 0
    fun dfs(n: TreeNode?, len: Int, dir: Int) {
        max = maxOf(max, len)
        if (n == null) return@dfs
        when (dir) {
            0 -> {
                dfs(n?.left, 0, -1)
                dfs(n?.right, 0, 1)
            }
            1 -> {
                dfs(n?.left, len + 1, -1)
                dfs(n?.right, 0, 1)
            }
            -1 -> {
                dfs(n?.right, len + 1, 1)
                dfs(n?.left, 0, -1)
            }
        }
    }
    dfs(root, 0, 0)
    return max
}

```

[blog post](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/solutions/3433418/kotlin-dfs/?orderBy=most_votes)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-19042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/185
#### Intuition
Search all the possibilities with DFS

#### Approach
Compute the `max` as you go
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$, for each level of `height` we traverse the full tree
- Space complexity:
$$O(log_2(n))$$

# 18.04.2023
[1768. Merge Strings Alternately](https://leetcode.com/problems/merge-strings-alternately/description/) easy

```kotlin

fun mergeAlternately(word1: String, word2: String): String =
(word1.asSequence().zip(word2.asSequence()) { a, b -> "$a$b" } +
word1.drop(word2.length) + word2.drop(word1.length))
.joinToString("")

```

[blog post](https://leetcode.com/problems/merge-strings-alternately/solutions/3429123/kotlin-sequence/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-18052023?sd=pf)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/184
#### Intuition
Do what is asked.
Handle the tail.
#### Approach
* we can use sequence `zip` operator
* for the tail, consider `drop`
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 17.04.2023
[1431. Kids With the Greatest Number of Candies](https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/description/) easy

```kotlin

fun kidsWithCandies(candies: IntArray, extraCandies: Int): List<Boolean> =
    candies.max()?.let { max ->
        candies.map { it + extraCandies >= max}
    } ?: listOf()

```

[blog post](https://leetcode.com/problems/kids-with-the-greatest-number-of-candies/solutions/3425529/kotlin-idiomatic/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-17042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/183
#### Intuition
We can just find the maximum and then try to add extra to every kid and check
#### Approach
Let's write the code
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 16.04.2023
[1639. Number of Ways to Form a Target String Given a Dictionary](https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/) hard

```kotlin

fun numWays(words: Array<String>, target: String): Int {
    val freq = Array(words[0].length) { LongArray(26) }
    for (i in 0..words[0].lastIndex)
    words.forEach { freq[i][it[i].toInt() - 'a'.toInt()]++ }

    val cache = Array(words[0].length) { LongArray(target.length) { -1L } }
    val m = 1_000_000_007L

    fun dfs(wpos: Int, tpos: Int): Long {
        if (tpos == target.length) return 1L
        if (wpos == words[0].length) return 0L
        if (cache[wpos][tpos] != -1L) return cache[wpos][tpos]
        val curr = target[tpos].toInt() - 'a'.toInt()
        val currFreq = freq[wpos][curr]
        val take = if (currFreq == 0L) 0L else
        dfs(wpos + 1, tpos + 1)
        val notTake = dfs(wpos + 1, tpos)
        val mul = (currFreq * take) % m
        val res = (mul + notTake) % m
        cache[wpos][tpos] = res
        return res
    }
    return dfs(0, 0).toInt()
}

```

[blog post](https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/solutions/3422184/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-16042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/182
#### Intuition
Consider an example: `bbc aaa ccc, target = ac`. We have 5 ways to form the `ac`:

```

// bbc aaa ccc   ac
//     a    c
//     a     c
//   c a
//      a    c
//   c  a

```

Looking at this, we deduce, that only count of every character at every position matter.

```

// 0 -> 1b 1a 1c
// 1 -> 1b 1a 1c
// 2 ->    1a 2c

```

To form `ac` we can start from position `0` or from `1`. If we start at `0`, we have one `c` at 1 plus two `c` at 2. And if we start at `1` we have two `c` at 3.
$$DP_{i,j} = Freq * DP_{i + 1, j + 1} + DP_{i + 1, j}$$

#### Approach
* precompute the `freq` array - count of each character at each position
* use an `Array` for faster cache
* use `long` to avoid overflow
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

# 15.04.2023
[2218. Maximum Value of K Coins From Piles](https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/description/) hard

```kotlin

fun maxValueOfCoins(piles: List<List<Int>>, k: Int): Int {
    val cache = Array(piles.size) { mutableListOf<Long>() }

        fun dfs(pile: Int, taken: Int): Long {
            if (taken >= k || pile >= piles.size) return 0
            if (cache[pile].size > taken) return cache[pile][taken]
            var max = dfs(pile + 1, taken)
            var sum = 0L
            for (j in 0..piles[pile].lastIndex) {
                val newTaken = taken + j + 1
                if (newTaken > k) break
                sum += piles[pile][j]
                max = maxOf(max, sum + dfs(pile + 1, newTaken))
            }
            cache[pile].add(max)
            return max
        }

        return dfs(0, 0).toInt()
    }

```

[blog post](https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/solutions/3418459/kotlin-dfs-cache/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-15042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/181
#### Intuition
Given the current pile, we can assume there is an optimal maximum value of the piles to the right of the current for every given number of `k`.
![leetcode_daily_backtrack.gif](https://assets.leetcode.com/users/images/9936598c-1906-47c8-ad95-cbb84a54ac32_1681537939.4210196.gif)

#### Approach
We can cache the result by the keys of every `pile to taken`

#### Complexity
- Time complexity:
$$O(kn^2)$$
- Space complexity:
$$O(kn^2)$$

# 14.04.2023
[516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/description/) medium

```kotlin

fun longestPalindromeSubseq(s: String): Int {
    // b + abcaba
    // b + ab_ab_
    // b + a_cab_
    // acbbc + a -> [acbbc]a x[from]==x[to]?1 + p[from+1][to-1]
    val p = Array(s.length) { Array(s.length) { 0 } }
    for (i in s.lastIndex downTo 0) p[i][i] = 1
    for (from in s.lastIndex - 1 downTo 0)
    for (to in from + 1..s.lastIndex)
    p[from][to] = if (s[from] == s[to]) {
        2 + if (to == from + 1) 0 else p[from + 1][to - 1]
    } else {
        maxOf(p[from][to - 1], p[from + 1][to])
    }
    return p[0][s.lastIndex]
}

```

[blog post](https://leetcode.com/problems/longest-palindromic-subsequence/solutions/3415189/kotlin-dp/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-14042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/180
#### Intuition
Simple DFS would not work as it will take $$O(2^n)$$ steps.
Consider the sequence: `acbbc` and a new element `a`. The already existing largest palindrome is `cbbc`. When adding a new element, we do not care about what is inside between `a..a`, just the largest value of it.
So, there is a DP equation derived from this observation: $$p[i][j] = eq ? 2 + p[i+1][j-1] : max(p[i][j-1], p[i+1][j])$$.
#### Approach
For cleaner code:
* precompute `p[i][i] = 1`
* exclude `0` and `lastIndex` from iteration
* start with `to = from + 1`
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

# 13.04.2023
[946. Validate Stack Sequences](https://leetcode.com/problems/validate-stack-sequences/description/) medium

```kotlin

fun validateStackSequences(pushed: IntArray, popped: IntArray): Boolean =
with(Stack<Int>()) {
    var pop = 0
    pushed.forEach {
        push(it)
        while (isNotEmpty() && peek() == popped[pop]) {
            pop()
            pop++
        }
    }
    isEmpty()
}

```

[blog post](https://leetcode.com/problems/validate-stack-sequences/solutions/3411131/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/13042023?sd=pf)
#### Telegram
https://t.me/leetcode_daily_unstoppable/179
#### Intuition
Do simulation using a Stack.
#### Approach
* use one iteration and a second pointer for `pop`
* empty the stack after inserting an element
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 12.04.2023
[71. Simplify Path](https://leetcode.com/problems/simplify-path/description/) medium

```kotlin

fun simplifyPath(path: String): String =
"/" + Stack<String>().apply {
    path.split("/").forEach {
        when (it) {
            ".." -> if (isNotEmpty()) pop()
            "." -> Unit
            "" -> Unit
            else -> push(it)
        }
    }
}.joinToString("/")

```

[blog post](https://leetcode.com/problems/simplify-path/solutions/3407165/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-12042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/178
#### Intuition
We can simulate what each of the `.` and `..` commands do by using a `Stack`.
#### Approach
* split the string by `/`
* add elements to the Stack if they are not commands and not empty
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 11.04.2023
[2390. Removing Stars From a String](https://leetcode.com/problems/removing-stars-from-a-string/description/) medium

```kotlin

fun removeStars(s: String): String = StringBuilder().apply {
    s.forEach {
        if (it == '*') setLength(length - 1)
        else append(it)
    }
}.toString()

```

[blog post](https://leetcode.com/problems/removing-stars-from-a-string/solutions/3402891/kotlin-stack/)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/177
#### Intuition
Iterate over a string. When `*` symbol met, remove last character, otherwise add it.
#### Approach
* we can use a `Stack`, or just `StringBuilder`
#### Complexity
- Time complexity:

$$O(n)$$

- Space complexity:

$$O(n)$$

# 10.04.2023
[20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/) medium

```

fun isValid(s: String): Boolean = with(Stack<Char>()) {
    val opened = hashSetOf('(', '[', '{')
    val match = hashMapOf(')' to '(' , ']' to '[', '}' to '{')
    !s.any { c ->
        when {
            c in opened -> false.also { push(c) }
            isEmpty() -> true
            else -> pop() != match[c]
        }
    } && isEmpty()
}

```

[blog post](https://leetcode.com/problems/valid-parentheses/solutions/3399214/kotlin-stack/)

#### Join me on Telegram

[telegram](https://t.me/leetcode_daily_unstoppable/176)

#### Intuition

Walk the string and push brackets to the stack. When bracket is closing, pop from it.
#### Approach
* use HashMap to check matching bracket.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 09.04.2023
[1857. Largest Color Value in a Directed Graph](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/description/) hard

[blog post](https://leetcode.com/problems/largest-color-value-in-a-directed-graph/solutions/3396443/kotlin-dfs-cache/)

```kotlin

fun largestPathValue(colors: String, edges: Array<IntArray>): Int {
    if (edges.isEmpty()) return if (colors.isNotEmpty()) 1 else 0
    val fromTo = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) -> fromTo.getOrPut(from) { mutableListOf() } += to }
        val cache = mutableMapOf<Int, IntArray>()
        var haveCycle = false
        fun dfs(curr: Int, visited: HashSet<Int> = HashSet()): IntArray {
            return cache.getOrPut(curr) {
                val freq = IntArray(26)
                if (visited.add(curr)) {
                    fromTo.remove(curr)?.forEach {
                        val childFreq = dfs(it, visited)
                        for (i in 0..25) freq[i] = maxOf(childFreq[i], freq[i])
                    }
                    freq[colors[curr].toInt() - 'a'.toInt()] += 1
                } else haveCycle = true
                freq
            }
        }
        var max = 0
        edges.forEach { (from, to) -> max = maxOf(max, dfs(from).max()!!) }
        return if (haveCycle) -1 else max
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/175
#### Intuition
![image.png](https://assets.leetcode.com/users/images/112cac51-7c3f-4d73-945e-58237ddb6ba5_1681022662.9757764.png)
![leetcode_daily_small.gif](https://assets.leetcode.com/users/images/36cddca8-50c2-4c8e-b5b6-317e30533a37_1681023914.0180423.gif)

For each node, there is only one answer of the maximum count of the same color. For each parent, $$c_p = max(freq_{child})+colors[curr]$$. We can cache the result and compute it using DFS and selecting maximum count from all the children.
#### Approach
* use `visited` set to detect cycles
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 08.04.2023
[133. Clone Graph](https://leetcode.com/problems/clone-graph/description/) medium

[blog post](https://leetcode.com/problems/clone-graph/solutions/3392609/kotlin-two-dfs/)

```kotlin

fun cloneGraph(node: Node?): Node? {
    if (node == null) return null
    val oldToNew = mutableMapOf<Node, Node>()
    fun dfs(n: Node) {
        if (oldToNew[n] == null) {
            oldToNew[n] = Node(n.`val`)
            n.neighbors.forEach {
                if (it != null) dfs(it)
            }
        }
    }
    fun dfs2(n: Node) {
        oldToNew[n]!!.apply {
            if (neighbors.isEmpty()) {
                n.neighbors.forEach {
                    if (it != null) {
                        neighbors.add(oldToNew[it])
                        dfs2(it)
                    }
                }
            }
        }
    }
    dfs(node)
    dfs2(node)
    return oldToNew[node]
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/174
#### Intuition
We can map every `old` node to its `new` node. Then one DFS for the creation, another for the linking.

#### Approach
* we can avoid using `visited` set by checking if a new node already has filled its neighbors.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 07.04.2023
[1020. Number of Enclaves](https://leetcode.com/problems/number-of-enclaves/description/) medium

[blog post](https://leetcode.com/problems/number-of-enclaves/solutions/3388636/kotlin-dfs/)

```kotlin

fun numEnclaves(grid: Array<IntArray>): Int {
    val visited = HashSet<Pair<Int, Int>>()
    fun dfs(x: Int, y: Int): Int {
        return if (x < 0 || y < 0 || x > grid[0].lastIndex || y > grid.lastIndex) 0
        else if (grid[y][x] == 1 && visited.add(x to y))
        1 + dfs(x - 1, y) + dfs(x + 1, y) + dfs(x, y - 1) + dfs(x, y + 1)
        else 0
    }
    for (y in 0..grid.lastIndex) {
        dfs(0, y)
        dfs(grid[0].lastIndex, y)
    }
    for (x in 0..grid[0].lastIndex) {
        dfs(x, 0)
        dfs(x, grid.lastIndex)
    }
    var count = 0
    for (y in 0..grid.lastIndex)
    for(x in 0..grid[0].lastIndex)
    count += dfs(x, y)
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/173
#### Intuition
Walk count all the `1` cells using DFS and a visited set.
#### Approach
We can use `visited` set, or modify the grid or use Union-Find.
To exclude the borders, we can visit them first with DFS.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

# 06.04.2023
[1254. Number of Closed Islands](https://leetcode.com/problems/number-of-closed-islands/description/) medium

[blog post](https://leetcode.com/problems/number-of-closed-islands/solutions/3385170/kotlin-dfs/)

```kotlin

fun closedIsland(grid: Array<IntArray>): Int {
    val visited = HashSet<Pair<Int, Int>>()
    val seen = HashSet<Pair<Int, Int>>()

    fun dfs(x: Int, y: Int): Boolean {
        seen.add(x to y)
        if (x >= 0 && y >= 0 && x < grid[0].size && y < grid.size
        && grid[y][x] == 0 &&  visited.add(x to y)) {
            var isBorder = x == 0 || y == 0 || x == grid[0].lastIndex || y == grid.lastIndex
            isBorder = dfs(x - 1, y) || isBorder
            isBorder = dfs(x, y - 1) || isBorder
            isBorder = dfs(x + 1, y) || isBorder
            isBorder = dfs(x, y + 1) || isBorder
            return isBorder
        }
        return false
    }

    var count = 0
    for (y in 0..grid.lastIndex)
    for (x in 0..grid[0].lastIndex)
    if (grid[y][x] == 0 && seen.add(x to y) && !dfs(x, y)) count++
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/172
#### Intuition
Use hint #1, if we don't count islands on the borders, we get the result. Now, just count all connected `0` cells that didn't connect to the borders. We can use DFS or Union-Find.
#### Approach
DFS will solve the problem.
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

# 05.04.2023
[2439. Minimize Maximum of Array](https://leetcode.com/problems/minimize-maximum-of-array/description/) medium

[blog post](https://leetcode.com/problems/minimize-maximum-of-array/solutions/3381720/kotlin-binary-search/)

```kotlin

fun minimizeArrayValue(nums: IntArray): Int {
    // 5 4 3 2 1 -> 5
    // 1 2 3 4 5 -> 3
    // 1 2 3 6 3
    // 1 2 6 3 3
    // 1 5 3 3 3
    // 3 3 3 3 3
    fun canArrangeTo(x: Long): Boolean {
        var diff = 0L
        for (i in nums.lastIndex downTo 0)
        diff = maxOf(0L, nums[i].toLong() - x + diff)
        return diff == 0L
    }
    var lo = 0
    var hi = Int.MAX_VALUE
    var min = Int.MAX_VALUE
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (canArrangeTo(mid.toLong())) {
            min = minOf(min, mid)
            hi = mid - 1
        } else lo = mid + 1
    }
    return min
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/171
#### Intuition
Observing the pattern, we can see, that any number from the `end` can be passed to the `start` of the array. One idea is to use two pointers, one pointing to the `biggest` value, another to the `smallest`. Given that biggest and smallest values changes, it will take $$O(nlog_2(n))$$ time to maintain such sorted structure.
Another idea, is that for any given `maximum value` we can walk an array from the end to the start and change values to be no bigger than it. This operation takes $$O(n)$$ time, and with the growth of the `maximum value` also increases a possibility to comply for all the elements. So, we can binary search in that space.
#### Approach
* careful with integers overflows
* for more robust binary search code:
* * check the final condition `lo == hi`
* * use inclusive `lo` and `hi`
* * always check the resulting value `min = minOf(min, mid)`
* * always move the borders `mid + 1` and `mid - 1`
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(1)$$

# 04.04.2023
[2405. Optimal Partition of String](https://leetcode.com/problems/optimal-partition-of-string/description/) medium

[blog post](https://leetcode.com/problems/optimal-partition-of-string/solutions/3377265/kotlin-bitmask/)

```kotlin

    var mask = 0
    fun partitionString(s: String): Int = 1 + s.count {
        val bit = 1 shl (it.toInt() - 'a'.toInt())
        (mask and bit != 0).also {
            if (it) mask = 0
            mask = mask or bit
        }
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/170
#### Intuition
Expand all the intervals until they met a duplicate character. This will be the optimal solution, as the minimum of the intervals correlates with the maximum of each interval length.
#### Approach
* use `hashset`, `[26]` array or simple `32-bit` mask to store visited flags for character
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 03.04.2023
[881. Boats to Save People](https://leetcode.com/problems/boats-to-save-people/description/) medium

[blog post](https://leetcode.com/problems/boats-to-save-people/solutions/3373007/kotlin-two-pointers/)

```kotlin

fun numRescueBoats(people: IntArray, limit: Int): Int {
    people.sort()
    var count = 0
    var lo = 0
    var hi = people.lastIndex
    while (lo <= hi) {
        if (lo < hi && people[hi] + people[lo] <= limit) lo++
        hi--
        count++
    }
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/169
#### Intuition
The optimal strategy comes from an intuition: for each `people[hi]` of a maximum weight, we can or can not add the one man `people[lo]` of a minimum weight.
#### Approach
Sort an array and move two pointers `lo` and `hi`.
* Careful with the ending condition, `lo == hi`
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(1)$$

# 02.04.2023
[2300. Successful Pairs of Spells and Potions](https://leetcode.com/problems/successful-pairs-of-spells-and-potions/description/) medium

[blog post](https://leetcode.com/problems/successful-pairs-of-spells-and-potions/solutions/3369146/kotlin-binary-search/)

```kotlin

fun successfulPairs(spells: IntArray, potions: IntArray, success: Long): IntArray {
    potions.sort()
    return IntArray(spells.size) { ind ->
        var lo = 0
        var hi = potions.lastIndex
        var minInd = potions.size
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (potions[mid].toLong() * spells[ind].toLong() >= success) {
                minInd = minOf(minInd, mid)
                hi = mid - 1
            } else lo = mid + 1
        }
        potions.size - minInd
    }
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/168
#### Intuition
If we sort `potions`, we can find the lowest possible value of `spell[i]*potion[i]` that is `>= success`. All other values are bigger by the math multiplication property.
#### Approach
* sort `potions`
* binary search the `lowest` index
* use `long` to solve the integer overflow
###### For more robust binary search code:
* use inclusive `lo` and `hi`
* do the last check `lo == hi`
* always compute the result `minInd`
* always move the `lo` and the `hi`
* safely compute `mid` to not overflow
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(n)$$

# 01.04.2023
[704. Binary Search](https://leetcode.com/problems/binary-search/description/) easy

[blog post](https://leetcode.com/problems/binary-search/solutions/3364415/kotlin-tricks/)

```kotlin

fun search(nums: IntArray, target: Int): Int {
    var lo = 0
    var hi = nums.lastIndex
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (nums[mid] == target) return mid
        if (nums[mid] < target) lo = mid + 1
        else hi = mid - 1
    }
    return -1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/167
#### Intuition
Just write binary search.

#### Approach
For more robust code:
* use including ranges `lo..hi`
* check the last condition `lo == hi`
* always check the exit condition `== target`
* compute `mid` without the integer overflow
* always move the boundary `mid + ` or `mid - 1`
* check yourself where to move the boundary, imagine moving closer to the `target`
#### Complexity
- Time complexity:
$$O(log_2(n))$$
- Space complexity:
$$O(1)$$

# 31.03.2023
[1444. Number of Ways of Cutting a Pizza](https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/description/) hard

[blog post](https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/solutions/3361348/kotlin-dfs-memo-prefixsum/)

```kotlin

data class Key(val x: Int, val y: Int, val c: Int)
fun ways(pizza: Array<String>, k: Int): Int {
    val havePizza = Array(pizza.size) { Array<Int>(pizza[0].length) { 0 } }

        val lastX = pizza[0].lastIndex
        val lastY = pizza.lastIndex
        for (y in lastY downTo 0) {
            var sumX = 0
            for (x in lastX downTo 0) {
                sumX += if (pizza[y][x] == 'A') 1 else 0
                havePizza[y][x] = sumX + (if (y == lastY) 0 else havePizza[y + 1][x])
            }
        }

        val cache = mutableMapOf<Key, Int>()
        fun dfs(x: Int, y: Int, c: Int): Int {
            return cache.getOrPut(Key(x, y, c)) {
                if (c == 0) return@getOrPut if (havePizza[y][x] > 0) 1 else 0
                var res = 0
                for (xx in x + 1..lastX) if (havePizza[y][x] > havePizza[y][xx])
                res = (res + dfs(xx, y, c - 1)) % 1_000_000_007

                for (yy in y + 1..lastY) if (havePizza[y][x] > havePizza[yy][x])
                res = (res + dfs(x, yy, c - 1)) % 1_000_000_007

                return@getOrPut res
            }
        }
        return dfs(0, 0, k - 1)
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/165
#### Intuition
The tricky problem is to how to program a number of cuts.
We can do the horizontal and vertical cuts decreasing available number `k` and tracking if we have any apples `before` and any apples `after` the cut. To track this, we can precompute a `prefix sum` of the apples, by each `top-left` corner to the end of the pizza. The stopping condition of the DFS is if we used all available cuts.

#### Approach
* carefully precompute prefix sum. You move by row, increasing `sumX`, then you move by column and reuse the result of the previous row.
* to detect if there are any apples above or to the left, compare the total number of apples precomputed from the start of the given `x,y` in the arguments and from the other side of the cut `xx,y` or `x, yy`.
#### Complexity
- Time complexity:
$$O(mnk(m+n))$$, mnk - number of cached states, (m+n) - search in each DFS step
- Space complexity:
$$O(mnk)$$

# 30.03.2023
[87. Scramble String](https://leetcode.com/problems/scramble-string/description/) hard

[blog post](https://leetcode.com/problems/scramble-string/solutions/3358175/kotlin-dfs-memo-no-substring/)

```kotlin

data class Key(val afrom: Int, val ato: Int, val bfrom: Int, val bto: Int)
fun isScramble(a: String, b: String): Boolean {
    val dp = HashMap<Key, Boolean>()
    fun dfs(key: Key): Boolean {
        return dp.getOrPut(key) {
            val (afrom, ato, bfrom, bto) = key
            val alength = ato - afrom
            val blength = bto - bfrom
            if (alength != blength) return@getOrPut false
            var same = true
            for (i in 0..alength)
            if (a[afrom + i] != b[bfrom + i]) same = false
            if (same) return@getOrPut true
            for (i in afrom..ato - 1) {
                if (dfs(Key(afrom, i, bfrom, bfrom + (i - afrom)))
                && dfs(Key(i + 1, ato, bfrom + (i - afrom) + 1, bto))) return@getOrPut true
                if (dfs(Key(afrom, i, bto - (i - afrom), bto))
                && dfs(Key(i + 1, ato, bfrom, bto - (i - afrom) - 1))) return@getOrPut true
            }

            return@getOrPut false
        }
    }
    return dfs(Key(0, a.lastIndex, 0, b.lastIndex))
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/164
#### Intuition
This is not a permutation's problem, as there are examples when we can't scramble two strings consisting of the same characters.
We can simulate the process and search the result using DFS.

#### Approach
A simple approach is to concatenate strings, but in Kotlin it gives TLE, so we need bottom up approach, or just operate with indices.
* use including indices ranges
* in Kotlin, don't forget `@getOrPut` when exiting lambda
#### Complexity
- Time complexity:
$$O(n^4)$$
- Space complexity:
$$O(n^4)$$

# 29.03.2023
[1402. Reducing Dishes](https://leetcode.com/problems/reducing-dishes/submissions/924018548/) hard

[blog post](https://leetcode.com/problems/reducing-dishes/solutions/3354056/kotlin-nlogn/)

```kotlin

fun maxSatisfaction(satisfaction: IntArray): Int {
    satisfaction.sort()
    var max = 0
    var curr = 0
    var diff = 0
    for (i in satisfaction.lastIndex downTo 0) {
        diff += satisfaction[i]
        curr += diff
        max = maxOf(max, curr)
    }

    return max
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/163
#### Intuition
Looking at the problem data examples, we intuitively deduce that the larger the number, the further it goes. We need to sort the array. With the negative numbers, we must compare all the results, excluding array prefixes.

#### Approach
The naive $$O(n^2)$$ solution will work. However, there is an optimal one if we simply go from the end.
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(n)$$

# 28.03.2023
[983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/description/) medium

[blog post](https://leetcode.com/problems/minimum-cost-for-tickets/solutions/3350465/kotlin-dfs-memo/)

```kotlin

fun mincostTickets(days: IntArray, costs: IntArray): Int {
    val cache = IntArray(days.size) { -1 }
    fun dfs(day: Int): Int {
        if (day >= days.size) return 0
        if (cache[day] != -1) return cache[day]
        var next = day
        while (next < days.size && days[next] - days[day] < 1) next++
        val costOne = costs[0] + dfs(next)
        while (next < days.size && days[next] - days[day] < 7) next++
        val costSeven = costs[1] + dfs(next)
        while (next < days.size && days[next] - days[day] < 30) next++
        val costThirty = costs[2] + dfs(next)
        return minOf(costOne, costSeven, costThirty).also { cache[day] = it}
    }
    return dfs(0)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/162
#### Intuition
For each day we can choose between tickets. Explore all of them and then choose minimum of the cost.

#### Approach
Let's write DFS with memoization algorithm as it is simple to understand.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 27.03.2023
[64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/description/) medium

[blog post](https://leetcode.com/problems/minimum-path-sum/solutions/3346543/kotlin-dfs-memo/)

```kotlin

    fun minPathSum(grid: Array<IntArray>): Int {
        val cache = mutableMapOf<Pair<Int, Int>, Int>()
        fun dfs(xy: Pair<Int, Int>): Int {
        return cache.getOrPut(xy) {
            val (x, y) = xy
            val curr = grid[y][x]
            if (x == grid[0].lastIndex && y == grid.lastIndex) curr else
            minOf(
            if (x < grid[0].lastIndex) curr + dfs((x + 1) to y)
            else Int.MAX_VALUE,
            if (y < grid.lastIndex) curr + dfs(x to (y + 1))
            else Int.MAX_VALUE
            )
        }
    }
    return dfs(0 to 0)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/161
##### Intuition
On each cell of the grid, there is only one minimum path sum. So, we can memorize it. Or we can use a bottom up DP approach.

#### Approach
Use DFS + memo, careful with the ending condition.

#### Complexity
- Time complexity:
$$O(n^2)$$, where $$n$$ - matrix size
- Space complexity:
$$O(n^2)$$

# 26.03.2023
[2360. Longest Cycle in a Graph](https://leetcode.com/problems/longest-cycle-in-a-graph/description/) hard

[blog post](https://leetcode.com/problems/longest-cycle-in-a-graph/solutions/3342651/kotlin-dfs/)

```kotlin

    fun longestCycle(edges: IntArray): Int {
        var maxLen = -1
        fun checkCycle(node: Int) {
            var x = node
            var len = 0
            do {
                if (x != edges[x]) len++
                x = edges[x]
            } while (x != node)
            if (len > maxLen) maxLen = len
        }

        val visited = HashSet<Int>()
        fun dfs(curr: Int, currPath: HashSet<Int>) {
            val isCurrentLoop = !currPath.add(curr)
            if (curr != -1 && !isCurrentLoop && visited.add(curr)) {
                dfs(edges[curr], currPath)
            } else if (curr != -1 && isCurrentLoop) checkCycle(curr)
        }
        for (i in 0..edges.lastIndex) dfs(i, HashSet<Int>())

        return maxLen
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/160
#### Intuition
We can walk all paths once and track the cycles with the DFS.

#### Approach
* Use separate visited sets for the current path and for the global visited nodes.
* Careful with `checkCycle` corner cases.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 25.03.2023
[2316. Count Unreachable Pairs of Nodes in an Undirected Graph](https://leetcode.com/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/description/) medium

[blog post](https://leetcode.com/problems/count-unreachable-pairs-of-nodes-in-an-undirected-graph/solutions/3338589/kotlin-union-find/)

```kotlin

fun countPairs(n: Int, edges: Array<IntArray>): Long {
    val uf = IntArray(n) { it }
    val sz = LongArray(n) { 1L }
    fun root(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun union(a: Int, b: Int) {
        val rootA = root(a)
        val rootB = root(b)
        if (rootA != rootB) {
            uf[rootB] = rootA
            sz[rootA] += sz[rootB]
            sz[rootB] = 0L
        }
    }
    edges.forEach { (from, to) -> union(from, to) }
    // 1 2 4 = 1*2 + 1*4 + 2*4 = 1*2 + (1+2)*4
    var sum = 0L
    var count = 0L
    sz.forEach { // 2 2 4 = 2*2 + 2*4 + 2*4 = 2*2 + (2+2)*4
        count += sum * it
        sum += it
    }
    return count
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/159
#### Intuition
To find connected components sizes, we can use Union-Find.
To count how many pairs, we need to derive the formula, observing the pattern. Assume we have groups sizes `3, 4, 5`, the number of pairs is the number of pairs between `3,4` + the number of pairs between `4,5` + between `3,5`. Or, $$count(a,b,c) = count(a,b) + count(b,c) + count(a,c) $$ where $$count(a,b) = a*b$$. So, $$count_{abc} = ab + bc + ac = ab + (a + b)c = count_{ab} + (a+b)c$$, or $$count_i = count_{i-1} + x_i*\sum_{j=0}^{i}x$$
#### Approach
* use path compression for better `root` time complexity
#### Complexity
- Time complexity:
$$O(height)$$
- Space complexity:
$$O(n)$$

# 24.03.2023
[1466. Reorder Routes to Make All Paths Lead to the City Zero](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/description/) medium

[blog post](https://leetcode.com/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/solutions/3334850/kotlin-bfs/)

```kotlin

    fun minReorder(n: Int, connections: Array<IntArray>): Int {
        val edges = mutableMapOf<Int, MutableList<Int>>()
        connections.forEach { (from, to) ->
            edges.getOrPut(from, { mutableListOf() }) += to
            edges.getOrPut(to, { mutableListOf() }) += -from
        }
        val visited = HashSet<Int>()
            var count = 0
            with(ArrayDeque<Int>().apply { add(0) }) {
                fun addNext(x: Int) {
                    if (visited.add(Math.abs(x))) {
                        add(Math.abs(x))
                        if (x > 0) count++
                    }
                }
                while (isNotEmpty()) {
                    repeat(size) {
                        val from = poll()
                        edges[from]?.forEach { addNext(it) }
                        edges[-from]?.forEach { addNext(it) }
                    }
                }
            }
            return count
        }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/158
#### Intuition
If our roads are undirected, the problem is simple: traverse with BFS from `0` and count how many roads are in the opposite direction.

#### Approach
We can use data structure or just use sign to encode the direction.
#### Complexity
- Time complexity:
$$O(V+E)$$
- Space complexity:
$$O(V+E)$$

# 23.03.2023
[1319. Number of Operations to Make Network Connected](https://leetcode.com/problems/number-of-operations-to-make-network-connected/description/) medium

[blog post](https://leetcode.com/problems/number-of-operations-to-make-network-connected/solutions/3331235/kotlin-union-find/)

```kotlin

fun makeConnected(n: Int, connections: Array<IntArray>): Int {
    var extraCables = 0
    var groupsCount = n
    val uf = IntArray(n) { it }
    fun findRoot(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun connect(a: Int, b: Int) {
        val rootA = findRoot(a)
        val rootB = findRoot(b)
        if (rootA == rootB) {
            extraCables++
            return
        }
        uf[rootB] = rootA
        groupsCount--
    }
    connections.forEach { (from, to) -> connect(from, to) }
    return if (extraCables < groupsCount - 1) -1 else groupsCount - 1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/157
#### Intuition
The number of cables we need is the number of disconnected groups of connected computers. Cables can be taken from the computers that have extra connections. We can do this using BFS/DFS and tracking visited set, counting extra cables if already visited node is in connection.
Another solution is to use Union-Find for the same purpose.

#### Approach
* for the better time complexity of the `findRoot` use path compression: `uf[x] = n`
#### Complexity
- Time complexity:
$$O(n*h)$$, $$h$$ - tree height, in a better implementation, can be down to constant. For Quick-Union-Find it is lg(n).
- Space complexity:
$$O(n)$$

# 22.03.2023
[2492. Minimum Score of a Path Between Two Cities](https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/description/) medium

[blog post](https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/solutions/3327604/kotlin-union-find/)

```kotlin

fun minScore(n: Int, roads: Array<IntArray>): Int {
    val uf = Array(n + 1) { it }
    val minDist = Array(n + 1) { Int.MAX_VALUE }
    fun findRoot(x: Int): Int {
        var n = x
        while (uf[n] != n) n = uf[n]
        uf[x] = n
        return n
    }
    fun union(a: Int, b: Int, dist: Int) {
        val rootA = findRoot(a)
        val rootB = findRoot(b)
        uf[rootB] = rootA
        minDist[rootA] = minOf(minDist[rootA], minDist[rootB], dist)
    }
    roads.forEach { (from, to, dist) -> union(from, to, dist) }
    return minDist[findRoot(1)]
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/156
#### Intuition
Observing the problem definition, we don't care about the path, but only about the minimum distance in a connected subset containing `1` and `n`. This can be solved by simple BFS, which takes $$O(V+E)$$ time and space. But ideal data structure for this problem is Union-Find.
* In an interview, it is better to just start with BFS, because explaining the time complexity of the `find` operation of Union-Find is difficult. https://algs4.cs.princeton.edu/15uf/

#### Approach
Connect all roads and update minimums in the Union-Find data structure. Use simple arrays for both connections and minimums.
* updating a root after finding it gives more optimal time
#### Complexity
- Time complexity:
$$O(E*tree_height)$$
- Space complexity:
$$O(n)$$

# 21.03.2023
[2348. Number of Zero-Filled Subarrays](https://leetcode.com/problems/number-of-zero-filled-subarrays/description/) medium

[blog post](https://leetcode.com/problems/number-of-zero-filled-subarrays/solutions/3323224/kotlin-count-of-subarrays/)

```kotlin

fun zeroFilledSubarray(nums: IntArray): Long {
    var currCount = 0L
    var sum = 0L
    nums.forEach {
        if (it == 0) currCount++ else currCount = 0L
        sum += currCount
    }
    return sum
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/155
#### Intuition
Consider the following sequence: `0`, `00`, `000`. Each time we are adding another element to the end of the previous. For `0` count of subarrays $$c_1 = 1$$, for `00` it is $$c_2 = c_1 + z_2$$, where $$z_2$$ is a number of zeros. So, the math equation is $$c_i = c_{i-1} + z_i$$, or $$c_n = \sum_{i=0}^{n}z_i $$

#### Approach
We can count subarray sums, then add them to the result, or we can just skip directly to adding to the result.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 20.03.2023
[605. Can Place Flowers](https://leetcode.com/problems/can-place-flowers/description/) easy

[blog post](https://leetcode.com/problems/can-place-flowers/solutions/3318756/kotlin-greedy/)

```kotlin

fun canPlaceFlowers(flowerbed: IntArray, n: Int): Boolean {
    var count = 0
    if (flowerbed.size == 1 && flowerbed[0] == 0) count++
    if (flowerbed.size >= 2 && flowerbed[0] == 0 && flowerbed[1] == 0) {
        flowerbed[0] = 1
        count++
    }
    for (i in 1..flowerbed.lastIndex - 1) {
        if (flowerbed[i] == 0 && flowerbed[i - 1] == 0 && flowerbed[i + 1] == 0) {
            flowerbed[i] = 1
            count++
        }
    }
    if (flowerbed.size >= 2 && flowerbed[flowerbed.lastIndex] == 0 && flowerbed[flowerbed.lastIndex - 1] == 0) count++
    return count >= n
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/154
#### Intuition
We can plant flowers greedily in every vacant place. This will be the maximum result because if we skip one item, the result is the same for even number of places or worse for odd.

#### Approach
* careful with corner cases
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 19.03.2023
[211. Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/description/) medium

[blog post](https://leetcode.com/problems/design-add-and-search-words-data-structure/solutions/3315405/kotlin-trie-queue/)

```kotlin

class Trie {
    val next = Array<Trie?>(26) { null }
    fun Char.ind() = toInt() - 'a'.toInt()
    operator fun get(c: Char): Trie? = next[c.ind()]
    operator fun set(c: Char, t: Trie) { next[c.ind()] = t }
    var isWord = false
}
class WordDictionary(val root: Trie = Trie()) {
    fun addWord(word: String) {
        var t = root
        word.forEach { t = t[it] ?: Trie().apply { t[it] = this } }
        t.isWord = true
    }

    fun search(word: String): Boolean = with(ArrayDeque<Trie>().apply { add(root) }) {
        !word.any { c ->
            repeat(size) {
                val t = poll()
                if (c == '.') ('a'..'z').forEach { t[it]?.let { add(it) } }
                else t[c]?.let { add(it) }
            }
            isEmpty()
        } && any { it.isWord }
    }
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/153
#### Intuition
We are already familiar with a `Trie` data structure, however there is a `wildcard` feature added. We have two options: add wildcard for every character in `addWord` method in $$O(w26^w)$$ time and then search in $$O(w)$$ time, or just add a word to `Trie` in $$O(w)$$ time and then search in $$O(w26^d)$$ time, where $$d$$ - is a wildcards count. In the description, there are at most `3` dots, so we choose the second option.

#### Approach
Let's try to write it in a Kotlin way, using as little words as possible.
#### Complexity
- Time complexity:
$$O(w)$$ add, $$O(w26^d)$$ search, where $$d$$ - wildcards count.
- Space complexity:
$$O(m)$$, $$m$$ - unique words suffixes count.

# 18.03.2023
[1472. Design Browser History](https://leetcode.com/problems/design-browser-history/description/) medium

[blog post](https://leetcode.com/problems/design-browser-history/solutions/3310280/kotlin-list/)

```kotlin

class BrowserHistory(homepage: String) {
    val list = mutableListOf(homepage)
    var curr = 0
    var last = 0

    fun visit(url: String) {
        curr++
        if (curr == list.size) {
            list.add(url)
        } else {
            list[curr] = url
        }
        last = curr
    }

    fun back(steps: Int): String {
        curr = (curr - steps).coerceIn(0, last)
        return list[curr]
    }

    fun forward(steps: Int): String {
        curr = (curr + steps).coerceIn(0, last)
        return list[curr]
    }

}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/152
#### Intuition
Simple solution with array list will work, just not very optimal for the memory.

#### Approach
Just implement it.
#### Complexity
- Time complexity:
$$O(1)$$ for all operations
- Space complexity:
$$O(n)$$, will keep all the links

# 17.03.2023
[208. Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/description/) medium

[blog post](https://leetcode.com/problems/implement-trie-prefix-tree/solutions/3306557/kotlin-just-implement-it/)

```kotlin

class Trie() {
    val root = Array<Trie?>(26) { null }
    fun Char.ind() = toInt() - 'a'.toInt()
    operator fun get(c: Char): Trie? = root[c.ind()]
    operator fun set(c: Char, v: Trie) { root[c.ind()] = v }
    var isWord = false

    fun insert(word: String) {
        var t = this
        word.forEach { t = t[it] ?: Trie().apply { t[it] = this} }
        t.isWord = true
    }

    fun String.search(): Trie? {
        var t = this@Trie
        forEach { t = t[it] ?: return@search null }
        return t
    }

    fun search(word: String) = word.search()?.isWord ?: false

    fun startsWith(prefix: String) = prefix.search() != null

}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/151
#### Intuition
Trie is a common known data structure and all must know how to implement it.

#### Approach
Let's try to write it Kotlin-way
#### Complexity
- Time complexity:
$$O(w)$$ access for each method call, where $$w$$ - is a word length
- Space complexity:
$$O(w*N)$$, where $$N$$ - is a unique words count.

# 16.03.2023
[106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/) medium

[blog post](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/solutions/3303076/kotlin-dfs/)

```kotlin

fun buildTree(inorder: IntArray, postorder: IntArray): TreeNode? {
    val inToInd = inorder.asSequence().mapIndexed { i, v -> v to i }.toMap()
    var postTo = postorder.lastIndex
    fun build(inFrom: Int, inTo: Int): TreeNode? {
        if (inFrom > inTo || postTo < 0) return null
        return TreeNode(postorder[postTo]).apply {
            val inInd = inToInd[postorder[postTo]]!!
            postTo--
            right = build(inInd + 1, inTo)
            left = build(inFrom, inInd - 1)
        }
    }
    return build(0, inorder.lastIndex)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/150
#### Intuition
Postorder traversal gives us the root of every current subtree. Next, we need to find this value in inorder traversal: from the left of it will be the left subtree, from the right - right.

#### Approach
* To more robust code, consider moving `postTo` variable as we go in the reverse-postorder: from the right to the left.
* store indices in a hashmap
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 15.03.2023
[958. Check Completeness of a Binary Tree](https://leetcode.com/problems/check-completeness-of-a-binary-tree/description/) medium

[blog post](https://leetcode.com/problems/check-completeness-of-a-binary-tree/solutions/3299207/kotlin-dfs/)

```kotlin

data class R(val min: Int, val max: Int, val complete: Boolean)
fun isCompleteTree(root: TreeNode?): Boolean {
    fun dfs(n: TreeNode): R {
        with(n) {
            if (left == null && right != null) return R(0, 0, false)
            if (left == null && right == null) return R(0, 0, true)
            val (leftMin, leftMax, leftComplete) = dfs(left)
            if (!leftComplete) return R(0, 0, false)
            if (right == null) return R(0, leftMax + 1, leftMin == leftMax && leftMin == 0)
            val (rightMin, rightMax, rightComplete) = dfs(right)
            if (!rightComplete) return R(0, 0, false)
            val isComplete = leftMin == rightMin && rightMin == rightMax
            || leftMin == leftMax && leftMin == rightMin + 1
            return R(1 + minOf(leftMin, rightMin), 1 + maxOf(leftMax, rightMax), isComplete)
        }
    }
    return root == null || dfs(root).complete
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/149
#### Intuition

![image.png](https://assets.leetcode.com/users/images/33007881-5b61-45c1-ab4b-fe7ec7852560_1678863559.1249547.png)

For each node, we can compute it's left and right child `min` and `max` depth, then compare them.
#### Approach
Right depth must not be larger than left.
There are no corner cases, just be careful.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(log_2(n))$$

# 14.03.2023
[129. Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/description/) medium

[blog post](https://leetcode.com/problems/sum-root-to-leaf-numbers/solutions/3295054/kotlin-dfs/)

```kotlin

fun sumNumbers(root: TreeNode?): Int = if (root == null) 0 else {
    var sum = 0
    fun dfs(n: TreeNode, soFar: Int) {
        with(n) {
            val x = soFar * 10 + `val`
            if (left == null && right == null) sum += x
            if (left != null) dfs(left, x)
            if (right != null) dfs(right, x)
        }
    }
    dfs(root, 0)

    sum
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/148
#### Intuition
Just make DFS and add to the sum if the node is a leaf.

#### Approach
The most trivial way is to keep `sum` variable outside the dfs function.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(log_2(n))$$

# 13.03.2023
[101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/description/) easy

[blog post](https://leetcode.com/problems/symmetric-tree/solutions/3291127/kotlin-bfs-recursion/)

```kotlin

data class H(val x: Int?)
fun isSymmetric(root: TreeNode?): Boolean {
    with(ArrayDeque<TreeNode>().apply { root?.let { add(it) } }) {
        while (isNotEmpty()) {
            val stack = Stack<H>()
                val sz = size
                repeat(sz) {
                    if (sz == 1 && peek().left?.`val` != peek().right?.`val`) return false
                    with(poll()) {
                        if (sz == 1 || it < sz / 2) {
                            stack.push(H(left?.`val`))
                            stack.push(H(right?.`val`))
                        } else {
                            if (stack.isEmpty() || stack.pop().x != left?.`val`) return false
                            if (stack.isEmpty() || stack.pop().x != right?.`val`) return false
                        }
                        left?.let { add(it)}
                        right?.let { add(it)}
                    }
                }
            }
        }
        return true
    }

    fun isSymmetric2(root: TreeNode?): Boolean {
        fun isSymmetric(leftRoot: TreeNode?, rightRoot: TreeNode?): Boolean {
            return leftRoot == null && rightRoot == null
            || leftRoot != null && rightRoot != null
            && leftRoot.`val` == rightRoot.`val`
            && isSymmetric(leftRoot.left, rightRoot.right)
            && isSymmetric(leftRoot.right, rightRoot.left)
        }
        return isSymmetric(root?.left, root?.right)
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/147
#### Intuition
Recursive solution based on idea that we must compare `left.left` with `right.right` and `left.right` with `right.left`.
Iterative solution is just BFS and Stack.

#### Approach
Recursive: just write helper function.
Iterative: save also `null`'s to solve corner cases.
#### Complexity
- Time complexity:
Recursive: $$O(n)$$
Iterative: $$O(n)$$
- Space complexity:
Recursive: $$O(log_2(n))$$
Iterative: $$O(n)$$

# 12.03.2023
[23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/description/) hard

[blog post](https://leetcode.com/problems/merge-k-sorted-lists/solutions/3287757/kotlin-pq-and-divide-and-conquer/)

```kotlin

    fun mergeKLists(lists: Array<ListNode?>): ListNode? {
        val root = ListNode(0)
        var curr: ListNode = root
        val pq = PriorityQueue<ListNode>(compareBy( { it.`val` }))
        lists.forEach { if (it != null) pq.add(it) }
        while (pq.isNotEmpty()) {
            val next = pq.poll()
            curr.next = next
            next.next?.let { pq.add(it) }
            curr = next
        }
        return root.next
    }
    fun mergeKLists2(lists: Array<ListNode?>): ListNode? {
        fun merge(oneNode: ListNode?, twoNode: ListNode?): ListNode? {
            val root = ListNode(0)
            var curr: ListNode = root
            var one = oneNode
            var two = twoNode
            while (one != null && two != null) {
                if (one.`val` <= two.`val`) {
                    curr.next = one
                    one = one.next
                } else {
                    curr.next = two
                    two = two.next
                }
                curr = curr.next!!
            }
            if (one != null) curr.next = one
            else if (two != null) curr.next = two

            return root.next
        }
        return lists.fold(null as ListNode?) { r, t -> merge(r, t) }
    }

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/146
#### Intuition
On each step, we need to choose a minimum from `k` variables. The best way to do this is to use `PriorityQeueu`
Another solution is to just iteratively merge the `result` to the next list from the array.

#### Approach
* use dummy head
For the `PriorityQueue` solution:
* use non-null values to more robust code
For the iterative solution:
* we can skip merging if one of the lists is empty
#### Complexity
- Time complexity:
* `PriorityQueue`: $$O(nlog(k))$$
* iterative merge: $$O(nk)$$
- Space complexity:
* `PriorityQueue`: $$O(k)$$
* iterative merge: $$O(1)$$

# 11.03.2023
[109. Convert Sorted List to Binary Search Tree](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/) medium

[blog post](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/solutions/3282962/kotlin-recursion/)

```kotlin

fun sortedListToBST(head: ListNode?): TreeNode? {
    if (head == null) return null
    if (head.next == null) return TreeNode(head.`val`)
    var one = head
    var twoPrev = head
    var two = head
    while (one != null && one.next != null) {
        one = one.next?.next
        twoPrev = two
        two = two?.next
    }
    twoPrev!!.next = null
    return TreeNode(two!!.`val`).apply {
        left = sortedListToBST(head)
        right = sortedListToBST(two!!.next)
    }
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/145
#### Intuition
One way is to convert linked list to array, then just build a binary search tree using divide and conquer technique. This will take $$O(nlog_2(n))$$ additional memory, and $$O(n)$$ time.
We can skip using the array and just compute the middle of the linked list each time.
#### Approach
Compute the middle of the linked list.
* careful with corner cases (check `fast.next != null` instead of `fast != null`)
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(log_2(n))$$ of additional space (for recursion)

# 10.03.2023
[382. Linked List Random Node](https://leetcode.com/problems/linked-list-random-node/description/) medium

[blog post](https://leetcode.com/problems/linked-list-random-node/solutions/3279169/kotlin-i-don-t-get-reservior-sampling-just-split-into-buckets-of-size-k/)

```kotlin

class Solution(val head: ListNode?) {
    val rnd = Random(0)
    var curr = head

    fun getRandom(): Int {
        val ind = rnd.nextInt(6)
        var peek: ListNode? = null
        repeat(6) {
            curr = curr?.next
            if (curr == null) curr = head
            if (it == ind) peek = curr
        }

        return peek!!.`val`
    }

}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/144

#### Intuition
Naive solution is trivial. For more interesting solution, you need to look at what others did on leetcode, read an article https://en.wikipedia.org/wiki/Reservoir_sampling and try to understand why it works.

My intuition was: if we need a probability `1/n`, where `n` - is a total number of elements, then what if we split all the input into buckets of size `k`, then choose from every bucket with probability `1/k`. It seems to work, but only for sizes starting from number `6` for the given input.
We just need to be sure, that number of `getRandom` calls are equal to number of buckets `n/k`.

#### Approach
Write the naive solution, then go to Wikipedia, and hope you will not get this in the interview.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 09.03.2023
[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/description/) medium

[blog post](https://leetcode.com/problems/linked-list-cycle-ii/solutions/3275105/kotlin-fast-and-slow-plus-trick/)

```kotlin

fun detectCycle(head: ListNode?): ListNode? {
    var one = head
    var two = head
    do {
        one = one?.next
        two = two?.next?.next
    } while (two != null && one != two)
    if (two == null) return null
    one = head
    while (one != two) {
        one = one?.next
        two = two?.next
    }
    return one
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/143
#### Intuition
![image.png](https://assets.leetcode.com/users/images/72ccd4d1-7aa6-40f1-ad87-86625f8e7241_1678342726.4682755.png)
There is a known algorithm to detect a cycle in a linked list. Move `slow` pointer one node at a time, and move `fast` pointer two nodes at a time. Eventually, if they meet, there is a cycle.
To know the connection point of the cycle, you can also use two pointers: one from where pointers were met, another from the start, and move both of them one node at a time until they meet.
How to derive this yourself?
* you can draw the diagram
* notice, when all the list is a cycle, nodes met at exactly where they are started
* meet point = cycle length + tail
#### Approach
* careful with corner cases.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 08.03.2023
[875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/description/) medium

[blog post](https://leetcode.com/problems/koko-eating-bananas/solutions/3271497/kotlin-binary-search/)

```kotlin

fun minEatingSpeed(piles: IntArray, h: Int): Int {
    fun canEatAll(speed: Long): Boolean {
        var time = 0L
        piles.forEach {
            time += (it.toLong() / speed) + if ((it.toLong() % speed) == 0L) 0L else 1L
        }
        return time <= h
    }
    var lo = 1L
    var hi = piles.asSequence().map { it.toLong() }.sum()!!
    var minSpeed = hi
    while (lo <= hi) {
        val speed = lo + (hi - lo) / 2
        if (canEatAll(speed)) {
            minSpeed = minOf(minSpeed, speed)
            hi = speed - 1
        } else {
            lo = speed + 1
        }
    }
    return minSpeed.toInt()
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/142
#### Intuition
Given the `speed` we can count how many `hours` take Coco to eat all the bananas. With growth of `speed` `hours` growth too, so we can binary search in that space.

#### Approach
For more robust binary search:
* use inclusive condition check `lo == hi`
* always move boundaries `mid + 1`, `mid - 1`
* compute the result on each step
#### Complexity
- Time complexity:
$$O(nlog_2(m))$$, `m` - is `hours` range
- Space complexity:
$$O(1)$$

# 07.03.2023
[2187. Minimum Time to Complete Trips](https://leetcode.com/problems/minimum-time-to-complete-trips/description/) medium

[blog post](https://leetcode.com/problems/minimum-time-to-complete-trips/solutions/3267486/kotlin-binary-search/)

```kotlin

fun minimumTime(time: IntArray, totalTrips: Int): Long {
    fun tripCount(timeGiven: Long): Long {
        var count = 0L
        for (t in time) count += timeGiven / t.toLong()
        return count
    }
    var lo = 0L
    var hi = time.asSequence().map { it.toLong() * totalTrips }.max()!!
    var minTime = hi
    while (lo <= hi) {
        val timeGiven = lo + (hi - lo) / 2
        val trips = tripCount(timeGiven)
        if (trips >= totalTrips) {
            minTime = minOf(minTime, timeGiven)
            hi = timeGiven - 1
        } else {
            lo = timeGiven + 1
        }
    }
    return minTime
}

```

#### Join me on telergam
https://t.me/leetcode_daily_unstoppable/140
#### Intuition
Naive approach is just to simulate the `time` running, but given the problem range it is not possible.
However, observing the `time` simulation results, we can notice, that by each `given time` there is a certain `number of trips`. And `number of trips` growths continuously with the growth of the `time`. This is a perfect condition to do a binary search in a space of the `given time`.
With `given time` we can calculate number of trips in a $$O(n)$$ complexity.

#### Approach
Do a binary search. For the `hi` value, we can peak a $$10^7$$ or just compute all the time it takes for every bus to trip.
For a more robust binary search:
* use inclusive `lo` and `hi`
* use inclusive check for the last case `lo == hi`
* compute the result on every step instead of computing it after the search
* always move the borders `mid + 1`, `mid - 1`

#### Complexity
- Time complexity:
$$O(nlog_2(m))$$, $$m$$ - is a time range
- Space complexity:
$$O(1)$$

# 06.03.2023
[1539. Kth Missing Positive Number](https://leetcode.com/problems/kth-missing-positive-number/description/) easy

[blog post](https://leetcode.com/problems/kth-missing-positive-number/solutions/3263077/kotlin-binary-search/)

```kotlin

fun findKthPositive(arr: IntArray, k: Int): Int {
    // 1 2 3 4 5 6 7 8 9 10 11
    // * 2 3 4 * * 7 * * *  11
    //   ^                  ^
    // 1 2 3 4 5
    // 2 3 4 7 11
    // 1
    //   1
    //     1
    //       3
    //         6
    //
    //       ^ 7 + (5-3) = 9
    //         arr[m] + (k-diff)
    //
    // 1 2
    // 7 8     k=1
    // 6
    //   6
    var lo = 0
    var hi = arr.lastIndex
    var res = -1
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        val diff = arr[mid] - mid - 1
        if (diff < k) {
            res = arr[mid] + (k - diff)
            lo = mid + 1
        } else {
            hi  = mid - 1
        }
    }
    return if (res == -1) k else res
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/139
#### Intuition
Let's observe an example:

```

// 1 2 3 4 5 6 7 8 9 10 11
// * 2 3 4 * * 7 * * *  11

```

For each number at its position, there are two conditions:
* if it stays in a correct position, then `num - pos == 0`
* if there is a missing number before it, then `num - pos == diff > 0`

We can observe the pattern and derive the formula for it:

```

// 1 2 3 4 5
// 2 3 4 7 11
// 1
//   1
//     1
//       3
//         6
//
//       ^ 7 + (5-3) = 9
//         arr[m] + (k-diff)

```

One corner case is if the missing numbers are at the beginning of the array:

```

// 1 2
// 7 8     k=1
// 6
//   6

```

Then the answer is just a `k`.
#### Approach
For more robust binary search code:
* use inclusive borders `lo` and `hi` (don't make of by 1 error)
* use inclusive last check `lo == hi` (don't miss one item arrays)
* always move the borders `mid + 1` or `mid - 1` (don't fall into an infinity loop)
* always compute the search if the case is `true` (don't compute it after the search to avoid mistakes)
#### Complexity
- Time complexity:
$$O(log_2(n))$$
- Space complexity:
$$O(n)$$

# 05.03.2023
[1345. Jump Game IV](https://leetcode.com/problems/jump-game-iv/description/) hard

[blog post](https://leetcode.com/problems/jump-game-iv/solutions/3259651/kotlin-bfs-pruning/)

```kotlin

fun minJumps(arr: IntArray): Int {
    val numToPos = mutableMapOf<Int, MutableList<Int>>()
        arr.forEachIndexed { i, n -> numToPos.getOrPut(n, { mutableListOf() }).add(i) }
        with(ArrayDeque<Int>().apply { add(0) }) {
            var jumps = 0
            val visited = HashSet<Int>()
                while(isNotEmpty()) {
                    repeat(size) {
                        val curr = poll()
                        if (curr == arr.lastIndex) return jumps
                        numToPos.remove(arr[curr])?.forEach { if (visited.add(it)) add(it) }
                        if (curr > 0 && visited.add(curr - 1)) add(curr - 1)
                        if (curr < arr.lastIndex && visited.add(curr + 1)) add(curr + 1)
                    }
                    jumps++
                }
            }
            return 0
        }

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/138
#### Intuition
Dynamic programming approach wouldn't work here, as we can tell from position `i` is it optimal before visiting both left and right subarrays.
Another way to find the shortest path is to just do Breath-First-Search.

#### Approach
This problem gives TLE until we do one trick:
* remove the visited nodes from the graph
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 04.03.2023
[2444. Count Subarrays With Fixed Bounds](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/description/) hard

[blog post](https://leetcode.com/problems/count-subarrays-with-fixed-bounds/solutions/3255030/kotlin-nlogn-but-not-tricky-solution-optimal/)

```kotlin

fun countSubarrays(nums: IntArray, minK: Int, maxK: Int): Long {
    val range = minK..maxK
    var i = 0
    var sum = 0L
    if (minK == maxK) {
        var count = 0
        for (i in 0..nums.lastIndex) {
            if (nums[i] == minK) count++
            else count = 0
            if (count > 0) sum += count
        }
        return sum
    }
    while (i < nums.size) {
        val curr = nums[i]
        if (curr in range) {
            val minInds = TreeSet<Int>()
                val maxInds = TreeSet<Int>()
                    var end = i
                    while (end < nums.size && nums[end] in range) {
                        if (nums[end] == minK) minInds.add(end)
                        else if (nums[end] == maxK) maxInds.add(end)
                        end++
                    }
                    if (minInds.size > 0 && maxInds.size > 0) {
                        var prevInd = i - 1
                        while (minInds.isNotEmpty() && maxInds.isNotEmpty()) {
                            val minInd = minInds.pollFirst()!!
                            val maxInd = maxInds.pollFirst()!!
                            val from = minOf(minInd, maxInd)
                            val to = maxOf(minInd, maxInd)
                            val remainLenAfter = (end - 1 - to).toLong()
                            val remainLenBefore = (from - (prevInd + 1)).toLong()
                            sum += 1L + remainLenAfter + remainLenBefore + remainLenAfter * remainLenBefore
                            prevInd = from
                            if (to == maxInd) maxInds.add(to)
                            else if (to == minInd) minInds.add(to)
                        }
                    }
                    if (i == end) end++
                    i = end
                } else i++
            }
            return sum
        }
and more clever solution:
fun countSubarrays(nums: IntArray, minK: Int, maxK: Int): Long {
    var sum = 0L
    if (minK == maxK) {
        var count = 0
        for (i in 0..nums.lastIndex) {
            if (nums[i] == minK) count++
            else count = 0
            if (count > 0) sum += count
        }
        return sum
    }
    val range = minK..maxK
    // 0 1 2 3 4 5 6 7 8 91011
    // 3 7 2 2 2 2 2 1 2 3 2 1
    //   b
    //               *...*
    //                   *...*
    var border = -1
    var posMin = -1
    var posMax = -1
    for (i in 0..nums.lastIndex) {
        when (nums[i]) {
            !in range -> border = i
            minK -> posMin = i
            maxK -> posMax = i
        }
        if (posMin > border && posMax > border)
        sum += minOf(posMin, posMax) - border
    }
    return sum
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/137
#### Intuition
First thought is that we can observe only subarrays, where all the elements are in a range `min..max`. Next, there are two possible scenarios:
1. If `minK==maxK`, our problem is a trivial count of the combinations, $$ 0 + 1 + .. + (n-1) + n = n*(n+1)/2$$
2. If `minK != maxK`, we need to take every `minK|maxK` pair, and count how many items are in range `before` them and how many `after`. Then, as we observe the pattern of combinations:

```

// 0 1 2 3 4 5 6    min=1, max=3
// ------------------
// 1 2 3 2 1 2 3
// 1 2 3          *** 0..2 remainLenAfter = 6 - 2 = 4
// 1 2 3 2
// 1 2 3 2 1
// 1 2 3 2 1 2
// 1 2 3 2 1 2 3
//     3 2 1      *** 2..4 remainLenAfter = 6 - 4 = 2
//     3 2 1 2
//     3 2 1 2 3
//   2 3 2 1               remainLenBefore = 2 - (0 + 1) = 1, sum += 1 + remainLenAfter += 1+2 += 3
//   2 3 2 1 2
//   2 3 2 1 2 3
//         1 2 3  *** 4..6 remainLenBefore = 4 - 4 + 1 = 1
//       2 1 2 3

// 1 2 1 2 3 2 3
// *.......*      *** 0..4 sum += 1 + 2 = 3
//     *...*      *** 2..4 rla = 6 - 4 = 2, rlb = 2 - (0 + 1) = 1, sum += 1 + rla + rlb + rlb*rla += 6 = 9

// 1 3 5 2 7 5
// *...*
//

```

we derive the formula: $$sum += 1 + suffix + prefix + suffix*prefix$$

A more clever, but less understandable solution: is to count how many times we take a condition where we have a `min` and a `max` and each time add `prefix` count. Basically, it is the same formula, but with a more clever way of computing. (It is like computing a combination sum by adding each time the counter to sum).
#### Approach

For the explicit solution, we take each interval, store positions of the `min` and `max` in a `TreeSet`, then we must take poll those mins and maxes and consider each range separately:

```

// 3 2 3 2 1 2 1
// *.......*
//     *...*

// 3 2 1 2 3 2 1
// *...*
//     *...*
//         *...*

// 3 2 1 2 1 2 3
// *...*
//     *.......*
//         *...*

// 3 2 1 2 3 3 3
// *...*
//     *...*

// 3 2 2 2 2 2 1
// *...........*

// 1 1 1 1 1 1 1
// *.*
//   *.*
//     *.*
//       *.*
//         *.*
//           *.*

```

For the tricky one solution, just see what other clever man already wrote on the leetcode site and hope you will not get the same problem in an interview.

#### Complexity

- Time complexity:
$$O(nlog_2(n))$$ -> $$O(n)$$

- Space complexity:
$$O(n)$$ -> $$O(1)$$

# 03.03.2023
[28. Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/) medium

[blog post](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/solutions/3250975/kotlin-rolling-hash/)

```kotlin

fun strStr(haystack: String, needle: String): Int {
    // f(x) = a + 32 * f(x - 1)
    // abc
    // f(a) = a + 0
    // f(ab) = b + 32 * (a + 0)
    // f(abc) = c + 32 * (b + 32 * (a + 0))
    //
    // f(b) = b + 0
    // f(bc) = c + 32 * (b + 0)
    //
    // f(abc) - f(bc) = 32^0*c + 32^1*b + 32^2*a - 32^0*c - 32^1*b = 32^2*a
    // f(bc) = f(abc) - 32^2*a
    var needleHash = 0L
    needle.forEach { needleHash = it.toLong() + 32L * needleHash }
    var currHash = 0L
    var pow = 1L
    repeat(needle.length) { pow *= 32L}
    for (curr in 0..haystack.lastIndex) {
        currHash = haystack[curr].toLong() + 32L * currHash
        if (curr >= needle.length)
        currHash -= pow * haystack[curr - needle.length].toLong()
        if (curr >= needle.lastIndex
        && currHash == needleHash
        && haystack.substring(curr - needle.lastIndex, curr + 1) == needle)
        return curr - needle.lastIndex
    }
    return -1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/136
#### Intuition
There is a `rolling hash` technique: you can compute hash for a sliding window using O(1) additional time.
Consider the math behind it:

```

// f(x) = a + 32 * f(x - 1)
// abc
// f(a) = a + 0
// f(ab) = b + 32 * (a + 0)
// f(abc) = c + 32 * (b + 32 * (a + 0))
//
// f(b) = b + 0
// f(bc) = c + 32 * (b + 0)
//
// f(abc) - f(bc) = 32^0*c + 32^1*b + 32^2*a - 32^0*c - 32^1*b = 32^2*a
// f(bc) = f(abc) - 32^2*a

```

Basically, you can subtract `char * 32^window_length` from the lower side of the sliding window.

#### Approach
* carefull with indexes
#### Complexity
- Time complexity:
$$O(n)$$, if our hash function is good, we good
- Space complexity:
$$O(n)$$, for substring, can be improved to O(1)

# 02.03.2023
[443. String Compression](https://leetcode.com/problems/string-compression/description/) medium

[blog post](https://leetcode.com/problems/string-compression/solutions/3246608/kotlin-contradiction-in-the-description/)

```kotlin

fun compress(chars: CharArray): Int {
    var end = 0
    var curr = 0
    while (curr < chars.size) {
        val c = chars[curr++]
        var currCount = 1
        while (curr < chars.size && c == chars[curr]) {
            curr++
            currCount++
        }
        chars[end++] = c
        if (currCount > 1) currCount.toString().forEach { chars[end++] = it }
    }
    return end
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/135
#### Intuition
You don't need to split a number into groups of `9`'s.
The right way to convert number `123` into a string is to divide it by 10 each time, then reverse a part of the array.

#### Approach
* Let's just do a naive `toString` for simplicity.
* to avoid mistakes with indexes, use explicit variable for count the duplicate chars
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(lg_10(n))$$, for storing `toString`. For this task it is a `4`

# 01.03.2023
[912. Sort an Array](https://leetcode.com/problems/sort-an-array/description/) medium

[blog post](https://leetcode.com/problems/sort-an-array/solutions/3242806/kotlin-quicksort/)

```kotlin

fun sortArray(nums: IntArray, from: Int = 0, to: Int = nums.lastIndex): IntArray {
    if (from >= to) return nums
    val mid = partition(nums, from, to)
    sortArray(nums, from, mid - 1)
    sortArray(nums, mid + 1, to)
    return nums
}
fun IntArray.swap(i: Int, j: Int) { this[i] = this[j].also { this[j] = this[i] } }
fun partition(nums: IntArray, from: Int, to: Int): Int {
    var border = nums[to]
    var afterBorder = from
    for (curr in from until to)
    if (nums[curr] < border) nums.swap(curr, afterBorder++)
    nums.swap(to, afterBorder)
    return afterBorder
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/134
#### Intuition
There are some tricks to optimize naive quicksort algorithm.
* choose between `lo`, `mid` and `hi` elements for the pivot instead of just `hi`
* shuffling the array before sorting
* starting with the smallest part of the array
* making the last recursion call with a `tailrec`
* sorting with `insertion sort` for a small parts

#### Approach
Let's just implement naive quicksort.
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(log_2(n))$$ for the recursion

# 28.02.2023
[652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/description/) medium

[blog post](https://leetcode.com/problems/find-duplicate-subtrees/solutions/3239077/kotlin-preorder-hashset/)

```kotlin

fun findDuplicateSubtrees(root: TreeNode?): List<TreeNode?> {
    val result = mutableListOf<TreeNode?>()
    val hashes = HashSet<String>()
        val added = HashSet<String>()
            fun hashDFS(node: TreeNode): String {
                return with(node) {
                    "[" + (left?.let { hashDFS(it) } ?: "*") +
                    "_" + `val` + "_" +
                    (right?.let { hashDFS(it) } ?: "*") + "]"
                }.also {
                    if (!hashes.add(it) && added.add(it)) result.add(node)
                }
            }
            if (root != null) hashDFS(root)
            return result
        }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/132
#### Intuition
We can traverse the tree and construct a hash of each node, then just compare nodes with equal hashes. Another way is to serialize the tree and compare that data.

#### Approach
Let's use pre-order traversal and serialize each node into string, also add that into `HashSet` and check for duplicates.
#### Complexity
- Time complexity:
$$O(n^2)$$, because of the string construction on each node.
- Space complexity:
$$O(n^2)$$

# 27.02.2023
[427. Construct Quad Tree](https://leetcode.com/problems/construct-quad-tree/description/) medium

[blog post](https://leetcode.com/problems/construct-quad-tree/solutions/3235370/kotlin-dfs/)

```kotlin

fun construct(grid: Array<IntArray>): Node? {
    if (grid.isEmpty()) return null
    fun dfs(xMin: Int, xMax: Int, yMin: Int, yMax: Int): Node? {
        if (xMin == xMax) return Node(grid[yMin][xMin] == 1, true)
        val xMid = xMin + (xMax - xMin) / 2
        val yMid = yMin + (yMax - yMin) / 2
        return Node(false, false).apply {
            topLeft = dfs(xMin, xMid, yMin, yMid)
            topRight = dfs(xMid + 1, xMax, yMin, yMid)
            bottomLeft = dfs(xMin, xMid, yMid + 1, yMax)
            bottomRight = dfs(xMid + 1, xMax, yMid + 1, yMax)
            if (topLeft!!.isLeaf && topRight!!.isLeaf
            && bottomLeft!!.isLeaf && bottomRight!!.isLeaf) {
                if (topLeft!!.`val` == topRight!!.`val`
                && topRight!!.`val` == bottomLeft!!.`val`
                && bottomLeft!!.`val` == bottomRight!!.`val`) {
                    `val` = topLeft!!.`val`
                    isLeaf = true
                    topLeft = null
                    topRight = null
                    bottomLeft = null
                    bottomRight = null
                }
            }
        }
    }
    return dfs(0, grid[0].lastIndex, 0, grid.lastIndex)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/131
#### Intuition
We can construct the tree using DFS and divide and conquer technique. Build four nodes, then check if all of them are equal leafs.

#### Approach
* use inclusive ranges to simplify the code
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 26.02.2023
[72. Edit Distance](https://leetcode.com/problems/edit-distance/description/) hard

[blog post](https://leetcode.com/problems/edit-distance/solutions/3231899/kotlin-dfs-memo/)

```kotlin

fun minDistance(word1: String, word2: String): Int {
    val dp = Array(word1.length + 1) { IntArray(word2.length + 1) { -1 } }
    fun dfs(i: Int, j: Int): Int {
        return when {
            dp[i][j] != -1 -> dp[i][j]
            i == word1.length && j == word2.length -> 0
            i == word1.length -> 1 + dfs(i, j+1)
            j == word2.length -> 1 + dfs(i+1, j)
            word1[i] == word2[j] -> dfs(i+1, j+1)
            else -> {
                val insert1Delete2 = 1 + dfs(i, j+1)
                val insert2Delete1 = 1 + dfs(i+1, j)
                val replace1Or2 = 1 + dfs(i+1, j+1)
                val res = minOf(insert1Delete2, insert2Delete1, replace1Or2)
                dp[i][j] = res
                res
            }
        }
    }
    return dfs(0, 0)
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/130
#### Intuition
Compare characters from each positions of the two strings. If they are equal, do nothing. If not, we can choose from three paths: removing, inserting or replacing. That will cost us `one` point of operations. Then, do DFS and choose the minimum of the operations.

#### Approach
Do DFS and use array for memoizing the result.
#### Complexity
- Time complexity:
$$O(n^2)$$, can be proven if you rewrite DP to bottom up code.
- Space complexity:
$$O(n^2)$$

# 25.02.2023
[121. Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/) easy

[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/solutions/3227923/kotlin-min-max/)

```kotlin

fun maxProfit(prices: IntArray): Int {
    var min = prices[0]
    var profit = 0
    prices.forEach {
        if (it < min) min = it
        profit = maxOf(profit, it - min)
    }
    return profit
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/129
#### Intuition
Max profit will be the difference between `max` and `min`. One thing to note, the `max` must follow after the `min`.

#### Approach
* we can just use current value as a `max` candidate instead of managing the `max` variable.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

# 24.02.2023
[1675. Minimize Deviation in Array](https://leetcode.com/problems/minimize-deviation-in-array/description/) hard

[blog post](https://leetcode.com/problems/minimize-deviation-in-array/solutions/3224614/kotlin-my-wrong-and-correct-intuition/)

```kotlin

fun minimumDeviation(nums: IntArray): Int {
    var minDiff = Int.MAX_VALUE
    with(TreeSet<Int>(nums.map { if (it % 2 == 0) it else it * 2 })) {
        do {
            val min = first()
            val max = pollLast()
            minDiff = minOf(minDiff, Math.abs(max - min))
            add(max / 2)
        } while (max % 2 == 0)
    }

    return minDiff
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/128
#### Intuition
We can notice, that the answer is the difference between the `min` and `max` from some resulting set of numbers.
My first (wrong) intuition was, that we can use two heaps for minimums and maximums, and only can divide by two from the maximum, and multiply by two from the minimum heap. That quickly transformed into too many edge cases.
The correct and tricky intuition: we can multiply all the numbers by 2, and then we can safely begin to divide all the maximums until they can be divided.

#### Approach
Use `TreeSet` to quickly access to the `min` and `max` elements.

#### Complexity
- Time complexity:
$$O(n(log_2(n) + log_2(h)))$$, where h - is a number's range
- Space complexity:
$$O(n)$$

# 23.02.2023
[502. IPO](https://leetcode.com/problems/ipo/description/) hard

[blog post](https://leetcode.com/problems/ipo/solutions/3221450/kotlin-wrong-and-correct-intuition/)

```kotlin

fun findMaximizedCapital(k: Int, w: Int, profits: IntArray, capital: IntArray): Int {
  val indices = Array(profits.size) { it }.apply { sortWith(compareBy( { capital[it] })) }
  var money = w
  with(PriorityQueue<Int>(profits.size, compareBy({ -profits[it] }))) {
    var i = 0
    repeat (k) {
      while (i <= indices.lastIndex && money >= capital[indices[i]]) add(indices[i++])
      if (isNotEmpty()) money += profits[poll()]
    }
  }
  return money
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/127
#### Intuition
My first (wrong) intuition: greedy add elements to the min-profit priority queue, then remove all low-profit elements from it, keeping essential items. It wasn't working, and the solution became too verbose.
Second intuition, after the hint: greedy add elements to the max-profit priority queue, then remove the maximum from it, which will be the best deal for the current money.

#### Approach
Sort items by increasing capital. Then, on each step, add all possible deals to the priority queue and take one best from it.

#### Complexity
- Time complexity:
  $$O(nlog_2(n))$$
- Space complexity:
  $$O(n)$$

# 22.02.2023
[1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/description/) medium

[blog post](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/solutions/3217409/kotlin-binary-search/)

```kotlin

fun shipWithinDays(weights: IntArray, days: Int): Int {
  var lo = weights.max()!!
  var hi = weights.sum()!!
  fun canShip(weight: Int): Boolean {
    var curr = 0
    var count = 1
    weights.forEach {
      curr += it
      if (curr > weight) {
        curr = it
        count++
      }
    }
    if (curr > weight) count++
    return count <= days
  }
  var min = hi
  while (lo <= hi) {
    val mid = lo + (hi - lo) / 2
    val canShip = canShip(mid)
    if (canShip) {
      min = minOf(min, mid)
      hi = mid - 1
    } else lo = mid + 1
  }
  return min
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/126
#### Intuition
Of all the possible capacities, there is an increasing possibility to carry the load. It may look like this: `not possible`, `not possible`, .., `not possible`, `possible`, `possible`, .., `possible`. We can binary search in that sorted space of possibilities.

#### Approach
To more robust binary search code:
* use inclusive `lo` and `hi`
* check the last case `lo == hi`
* check target condition separately `min = minOf(min, mid)`
* always move boundaries `lo` and `hi`
#### Complexity
- Time complexity:
  $$O(nlog_2(n))$$
- Space complexity:
  $$O(1)$$
 
# 21.02.2023
[540. Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/description/) medium

[blog post](https://leetcode.com/problems/single-element-in-a-sorted-array/solutions/3213551/kotlin-odd-even-positions-binary-search/)

```kotlin

fun singleNonDuplicate(nums: IntArray): Int {
    var lo = 0
    var hi = nums.lastIndex
    // 0 1 2 3 4
    // 1 1 2 3 3
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        val prev = if (mid > 0) nums[mid-1] else -1
        val next = if (mid < nums.lastIndex) nums[mid+1] else Int.MAX_VALUE
        val curr = nums[mid]
        if (prev < curr && curr < next) return curr

        val oddPos = mid % 2 != 0
        val isSingleOnTheLeft = oddPos && curr == next || !oddPos && curr == prev

        if (isSingleOnTheLeft) hi = mid - 1 else lo = mid + 1
    }
    return -1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/125
#### Intuition
This problem is a brain-teaser until you notice that pairs are placed at `even-odd` positions before the target and at `odd-even` positions after.
#### Approach
Let's write a binary search. For more robust code, consider:
* use inclusive `lo` and `hi`
* always move `lo` or `hi`
* check for the target condition and return early
#### Complexity
- Time complexity:
$$O(log_2(n))$$
- Space complexity:
$$O(1)$$

# 20.02.2023
[35. Search Insert Position](https://leetcode.com/problems/search-insert-position/description/) easy

[blog post](https://leetcode.com/problems/search-insert-position/solutions/3208831/kotlin-binary-search/)

```kotlin

    fun searchInsert(nums: IntArray, target: Int): Int {
        var lo = 0
        var hi = nums.lastIndex
        while (lo <= hi) {
            val mid = lo + (hi - lo) / 2
            if (target == nums[mid]) return mid
            if (target > nums[mid]) lo = mid + 1
            else hi = mid - 1
        }
        return lo
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/124
#### Intuition
Just do a binary search

#### Approach
For more robust code consider:
* use only inclusive boundaries `lo` and `hi`
* loop also the last case when `lo == hi`
* always move boundaries `mid + 1` or `mid - 1`
* use distinct check for the exact match `nums[mid] == target`
* return `lo` position - this is an insertion point

#### Complexity
- Time complexity:
  $$O(log_2(n))$$
- Space complexity:
  $$O(1)$$

# 19.02.2023
[103. Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/) medium

[blog post](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/solutions/3204919/kotlin-bfs/)

```kotlin 
    fun zigzagLevelOrder(root: TreeNode?): List<List<Int>> = mutableListOf<List<Int>>().also { res ->
            with(ArrayDeque<TreeNode>().apply { root?.let { add(it) } }) {
                while (isNotEmpty()) {
                    val curr = LinkedList<Int>().apply { res.add(this) }
                    repeat(size) {
                        with(poll()) {
                            with(curr) { if (res.size % 2 == 0) addFirst(`val`) else addLast(`val`) }
                            left?.let { add(it) }
                            right?.let { add(it) }
                        }
                    }
                }
            }
        }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/123
#### Intuition
Each BFS step gives us a level, which one we can reverse if needed.

#### Approach
* for zigzag, we can skip a boolean variable and track result count.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

# 18.02.2023
[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/) easy

[blog post](https://leetcode.com/problems/invert-binary-tree/solutions/3200281/kotlin-one-liner/)

```kotlin 
    fun invertTree(root: TreeNode?): TreeNode? = 
        root?.apply { left = invertTree(right).also { right = invertTree(left) } }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/122
#### Intuition
Walk tree with Depth-First Search and swap each left and right nodes.
#### Approach
Let's write a recursive one-liner.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(log_2(n))$$

# 17.02.2023
[783. Minimum Distance Between BST Nodes](https://leetcode.com/problems/minimum-distance-between-bst-nodes/submissions/899622255/) easy

[blog post](https://leetcode.com/problems/minimum-distance-between-bst-nodes/solutions/3196399/kotlin-morris-traversal/)

```kotlin 
    fun minDiffInBST(root: TreeNode?): Int {
        var prev: TreeNode? = null
        var curr = root
        var minDiff = Int.MAX_VALUE
        while (curr != null) {
            if (curr.left == null) {
                if (prev != null) minDiff = minOf(minDiff, Math.abs(curr.`val` - prev.`val`))
                prev = curr
                curr = curr.right
            } else {
                var right = curr.left!!
                while (right.right != null && right.right != curr) right = right.right!!
                if (right.right == curr) {
                    right.right = null
                    curr = curr.right
                } else {
                    right.right = curr
                    val next = curr.left
                    curr.left = null
                    curr = next
                }
            }
        }
        return minDiff
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/121
#### Intuition
Given that this is a Binary Search Tree, `inorder` traversal will give us an increasing sequence of nodes. Minimum difference will be one of the adjacent nodes differences.
#### Approach
Let's write Morris Traversal. Store current node at the rightmost end of the left children.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$
 
# 16.02.2023
[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/) easy

[blog post](https://leetcode.com/problems/maximum-depth-of-binary-tree/solutions/3192288/kotlin-one-liner/)

```kotlin 
    fun maxDepth(root: TreeNode?): Int =
        root?.run { 1 + maxOf(maxDepth(left), maxDepth(right)) } ?: 0

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/120
#### Intuition
Do DFS and choose the maximum on each step.

#### Approach
Let's write a one-liner.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(log_2(n))$$

# 15.02.2023
[989. Add to Array-Form of Integer](https://leetcode.com/problems/add-to-array-form-of-integer/description/) easy

[blog post](https://leetcode.com/problems/add-to-array-form-of-integer/solutions/3188017/kotlin-single-pass/)

```kotlin 
    fun addToArrayForm(num: IntArray, k: Int): List<Int> {
        var carry = 0
        var i = num.lastIndex
        var n = k
        val res = LinkedList<Int>()
        while (i >= 0 || n > 0 || carry > 0) {
            val d1 = if (i >= 0) num[i--] else 0
            val d2 = if (n > 0) n % 10 else 0
            var d = d1 + d2 + carry
            res.addFirst(d % 10)
            carry = d / 10 
            n = n / 10
        }
        return res
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/119
#### Intuition
Iterate from the end of the array and calculate sum of `num % 10`, `carry` and `num[i]`.

#### Approach
* use linked list to add to the front of the list in O(1)
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$
 
# 14.02.2023
[67. Add Binary](https://leetcode.com/problems/add-binary/description/) easy

[blog post](https://leetcode.com/problems/add-binary/solutions/3183889/kotlin-build-string/)

```kotlin 
        fun addBinary(a: String, b: String): String = StringBuilder().apply {
        var o = 0
        var i = a.lastIndex
        var j = b.lastIndex
        while (i >= 0 || j >= 0 || o == 1) {
            var num = o
            o = 0
            if (i >= 0 && a[i--] == '1') num++
            if (j >= 0 && b[j--] == '1') num++
            when (num) {
                0 -> append('0')
                1 -> append('1')
                2 -> {
                    append('0')
                    o = 1
                }
                else -> {
                    append('1')
                    o = 1
                }
            }
        }
    }.reverse().toString()

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/118
#### Intuition
Scan two strings from the end and calculate the result.

#### Approach
* keep track of the overflow
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

# 13.02.2023
[1523. Count Odd Numbers in an Interval Range](https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/description/) easy

[blog post](https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/solutions/3179265/kotlin-o-1/)

```kotlin 
    fun countOdds(low: Int, high: Int): Int {
        if (low == high) return if (low % 2 == 0) 0 else 1
        val lowOdd = low % 2 != 0
        val highOdd = high % 2 != 0
        val count = high - low + 1
        return if (lowOdd && highOdd) {
            1 + count / 2
        } else if (lowOdd || highOdd) {
            1 + (count - 1) / 2
        } else {
            1 + ((count - 2) / 2)
        }
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/117
#### Intuition
Count how many numbers in between, subtract even on the start and the end, then divide by 2.

#### Complexity
- Time complexity:
  $$O(1)$$
- Space complexity:
  $$O(1)$$

# 12.02.2023
[2477. Minimum Fuel Cost to Report to the Capital](https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital/description/) medium

[blog post](https://leetcode.com/problems/minimum-fuel-cost-to-report-to-the-capital/solutions/3175457/kotlin-dfs-with-picture/)

```kotlin 
    data class R(val cars: Long, val capacity: Int, val fuel: Long)
    fun minimumFuelCost(roads: Array<IntArray>, seats: Int): Long {
        val nodes = mutableMapOf<Int, MutableList<Int>>()
        roads.forEach { (from, to) ->
            nodes.getOrPut(from, { mutableListOf() }) += to
            nodes.getOrPut(to, { mutableListOf() }) += from
        }
        fun dfs(curr: Int, parent: Int): R {
            val children = nodes[curr]
            if (children == null) return R(1L, seats - 1, 0L)
            var fuel = 0L
            var capacity = 0
            var cars = 0L
            children.filter { it != parent }.forEach {
                val r = dfs(it, curr)
                fuel += r.cars + r.fuel
                capacity += r.capacity
                cars += r.cars
            }
            // seat this passenger
            if (capacity == 0) {
                cars++
                capacity = seats - 1
            } else capacity--
            // optimize cars
            while (capacity - seats >= 0) {
                capacity -= seats
                cars--
            }
            return R(cars, capacity, fuel)
        }
        return dfs(0, 0).fuel
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/116
#### Intuition

![image.png](https://assets.leetcode.com/users/images/bc400a1d-8fae-4bf5-bc93-3ad1eca4737c_1676194010.773879.png)

Let's start from each leaf (node without children). We give `one` car, `seats-1` capacity and `zero` fuel. When children cars arrive, each of them consume `cars` capacity of the fuel. On the hub (node with children), we sat another one passenger, so `capacity--` and we can optimize number of cars arrived, if total `capacity` is more than one car `seats` number.
#### Approach
Use DFS and data class for the result.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(h)$$, h - height of the tree, can be `0..n`
 
# 11.02.2023
[1129. Shortest Path with Alternating Colors](https://leetcode.com/problems/shortest-path-with-alternating-colors/description/) medium

[blog post](https://leetcode.com/problems/shortest-path-with-alternating-colors/solutions/3171245/kotlin-just-bfs/)

```kotlin 
    fun shortestAlternatingPaths(n: Int, redEdges: Array<IntArray>, blueEdges: Array<IntArray>): IntArray {
        val edgesRed = mutableMapOf<Int, MutableList<Int>>()
        val edgesBlue = mutableMapOf<Int, MutableList<Int>>()
        redEdges.forEach { (from, to) ->
            edgesRed.getOrPut(from, { mutableListOf() }).add(to)
        }
        blueEdges.forEach { (from, to) ->
            edgesBlue.getOrPut(from, { mutableListOf() }).add(to)
        }
        val res = IntArray(n) { -1 }
        val visited = hashSetOf<Pair<Int, Boolean>>()
        var dist = 0
        with(ArrayDeque<Pair<Int, Boolean>>()) {
            add(0 to true)
            add(0 to false)
            visited.add(0 to true)
            visited.add(0 to false)
            while (isNotEmpty()) {
                repeat(size) {
                    val (node, isRed) = poll()
                    if (res[node] == -1 || res[node] > dist) res[node] = dist
                    val edges = if (isRed) edgesRed else edgesBlue
                    edges[node]?.forEach {
                        if (visited.add(it to !isRed)) add(it to !isRed)
                    }
                }
                dist++
            }
        }
        return res
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/115
#### Intuition
We can calculate all the shortest distances in one pass BFS.
#### Approach
Start with two simultaneous points, one for red and one for blue. Keep track of the color.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$
 
# 10.02.2023
[1162. As Far from Land as Possible](https://leetcode.com/problems/as-far-from-land-as-possible/description/) medium

[blog post](https://leetcode.com/problems/as-far-from-land-as-possible/solutions/3167082/kotlin-bfs/)

```kotlin 
    fun maxDistance(grid: Array<IntArray>): Int = with(ArrayDeque<Pair<Int, Int>>()) {
        val n = grid.size
        val visited = hashSetOf<Pair<Int, Int>>()
        fun tryAdd(x: Int, y: Int) {
            if (x < 0 || y < 0 || x >= n || y >= n) return
            (x to y).let { if (visited.add(it)) add(it) }
        }
        for (yStart in 0 until n)
            for (xStart in 0 until n) 
                if (grid[yStart][xStart] == 1) tryAdd(xStart, yStart)
        if (size == n*n) return -1
        var dist = -1
        while(isNotEmpty()) {
            repeat(size) {
                val (x, y) = poll()
                tryAdd(x-1, y)
                tryAdd(x, y-1)
                tryAdd(x+1, y)
                tryAdd(x, y+1)
            }
            dist++
        }
        dist
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/114
#### Intuition
Let's do a wave from each land and wait until all the last water cell reached. This cell will be the answer.
#### Approach
Add all land cells into BFS, then just run it.
#### Complexity
- Time complexity:
  $$O(n^2)$$
- Space complexity:
  $$O(n^2)$$

# 9.02.2023
[2306. Naming a Company](https://leetcode.com/problems/naming-a-company/description/) hard

[blog post](https://leetcode.com/problems/naming-a-company/solutions/3163405/kotlin-intersect-suffix-buckets/)

```kotlin 
    fun distinctNames(ideas: Array<String>): Long {
        // c -> offee
        // d -> onuts
        // t -> ime, offee
        val prefToSuf = Array(27) { hashSetOf<String>() }
        for (idea in ideas)
            prefToSuf[idea[0].toInt() - 'a'.toInt()].add(idea.substring(1, idea.length))
        var count = 0L
        for (i in 0..26) 
            for (j in i + 1..26) 
                count += prefToSuf[i].count { !prefToSuf[j].contains(it) } * prefToSuf[j].count { ! prefToSuf[i].contains(it) }
        return count * 2L
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/113
#### Intuition
If we group ideas by the suffixes and consider only the unique elements, the result will be the intersection of the sizes of the groups. (To deduce this you must sit and draw, or have a big brain, or just use a hint)

#### Approach
Group and multiply. Don't forget to remove repeating elements in each two groups.
#### Complexity
- Time complexity:
  $$O(26^2n)$$
- Space complexity:
  $$O(n)$$
 
# 8.02.2023
[45. Jump Game II](https://leetcode.com/problems/jump-game-ii/description/) medium

[blog post](https://leetcode.com/problems/jump-game-ii/solutions/3161513/kotlin-greedy-stack/)

```kotlin 
    fun jump(nums: IntArray): Int {
        if (nums.size <= 1) return 0
        val stack = Stack<Int>()
        // 0 1 2 3 4 5 6 7 8 9 1011121314
        // 7 0 9 6 9 6 1 7 9 0 1 2 9 0 3
        //                             *
        //                           *
        //                         * * *
        //                       * * *
        //                     * *
        //                   *    
        //                 * * * * * * *
        //               * * * * * * * *
        //             * *
        //           * * * * * * *
        //         * * * * * * * * * *
        //       * * * * * * *
        //     * * * * * * * * * *
        //   *
        // * * * * * * * *
        // 3 4 3 2 5 4 3
        //             *
        //           * *
        //         * * *
        //       * * *
        //     * * * *
        //   * * * * *
        // * * * *
        // 0 1 2 3 4 5 6 7 8 9 1011
        // 5 9 3 2 1 0 2 3 3 1 0 0
        //                       *
        //                     *
        //                   * *
        //                 * * * *
        //               * * * *
        //             * * *
        //           *
        //         * *
        //       * * *
        //     * * * *
        //   * * * * * * * * * *
        // * * * * * *
        for (pos in nums.lastIndex downTo 0) {
            var canReach = minOf(pos + nums[pos], nums.lastIndex)
            if (canReach == nums.lastIndex) stack.clear()
            while (stack.size > 1 && stack.peek() <= canReach) {
                val top = stack.pop()
                if (stack.peek() > canReach) {
                    stack.push(top)
                    break
                }
            }
            stack.push(pos)
        }
        return stack.size
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/112
#### Intuition
The dynamic programming solution is trivial, and can be done in $$O(n^2)$$.
Greedy solution is to scan from back to front and keep only jumps that starts after the current max jump.

#### Approach
* use stack to store jumps
* pop all jumps less than current `maxReach`
* pop all except the last that can reach, so don't break the sequence.

#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$
 
# 7.02.2023
[904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/description/) medium

[blog post](https://leetcode.com/problems/fruit-into-baskets/solutions/3154719/kotlin-greedy/)

```kotlin 
    fun totalFruit(fruits: IntArray): Int {
        if (fruits.size <= 2) return fruits.size
        var type1 = fruits[fruits.lastIndex]
        var type2 = fruits[fruits.lastIndex - 1]
        var count = 2
        var max = 2
        var prevType = type2
        var prevTypeCount = if (type1 == type2) 2 else 1
        for (i in fruits.lastIndex - 2 downTo 0) {
            val type = fruits[i]
            if (type == type1 || type == type2 || type1 == type2) {
                if (type1 == type2 && type != type1) type2 = type
                if (type == prevType) prevTypeCount++
                else prevTypeCount = 1
                count++
            } else {
                count = prevTypeCount + 1
                type2 = type
                type1 = prevType
                prevTypeCount = 1
            }
            max = maxOf(max, count)
            prevType = type
        }
        return max
    }

```

#### Join daily telegram
https://t.me/leetcode_daily_unstoppable/111
#### Intuition
We can scan fruits linearly from the tail and keep only two types of fruits.
#### Approach
* careful with corner cases
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$

# 6.02.2023
[1470. Shuffle the Array](https://leetcode.com/problems/shuffle-the-array/description/) easy

[blog post](https://leetcode.com/problems/shuffle-the-array/solutions/3151995/kotlin-two-pointers-o-n-space/)

```kotlin 
    fun shuffle(nums: IntArray, n: Int): IntArray {
        val arr = IntArray(nums.size)
        var left = 0
        var right = n
        var i = 0
        while (i < arr.lastIndex) {
            arr[i++] = nums[left++]
            arr[i++] = nums[right++]
        }
        return arr
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/110
#### Intuition
Just do what is asked.
#### Approach
For simplicity, use two pointers for the source, and one for the destination.
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

# 5.02.2023
[438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/) medium

[blog post](https://leetcode.com/problems/find-all-anagrams-in-a-string/solutions/3145307/kotlin-frequencies/)

```kotlin 
    fun findAnagrams(s: String, p: String): List<Int> {
        val freq = IntArray(26) { 0 }
        var nonZeros = 0
        p.forEach { 
            val ind = it.toInt() - 'a'.toInt()
            if (freq[ind] == 0) nonZeros++
            freq[ind]--
        }
        val res = mutableListOf<Int>()
        for (i in 0..s.lastIndex) {
            val currInd = s[i].toInt() - 'a'.toInt()
            if (freq[currInd] == 0) nonZeros++
            freq[currInd]++
            if (freq[currInd] == 0) nonZeros--
            if (i >= p.length) {
                val ind = s[i - p.length].toInt() - 'a'.toInt()
                if (freq[ind] == 0) nonZeros++
                freq[ind]--
                if (freq[ind] == 0) nonZeros--
            }
            if (nonZeros == 0) res += i - p.length + 1
        }
        return res
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/109
#### Intuition
We can count frequencies of `p` and then scan `s` to match them.

#### Approach
* To avoid checking a frequencies arrays, we can count how many frequencies are not matching, and add only when non-matching count is zero.
#### Complexity
- Time complexity:
  $$O(n)$$

- Space complexity:
  $$O(1)$$
 
# 4.02.2023
[567. Permutation in String](https://leetcode.com/problems/permutation-in-string/description/) medium

[blog post](https://leetcode.com/problems/permutation-in-string/solutions/3139851/kotlin-frequencies/?orderBy=most_votes)

```kotlin 
    fun checkInclusion(s1: String, s2: String): Boolean {
        val freq1 = IntArray(26) { 0 }
        s1.forEach {  freq1[it.toInt() - 'a'.toInt()]++  }
        val freq2 = IntArray(26) { 0 }
        for (i in 0..s2.lastIndex) {
            freq2[s2[i].toInt() - 'a'.toInt()]++
            if (i >= s1.length) freq2[s2[i - s1.length].toInt() - 'a'.toInt()]--
            if (Arrays.equals(freq1, freq2)) return true
        }
        return false
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/108
#### Intuition
We can count the chars frequencies in the `s1` string and use the sliding window technique to count and compare char frequencies in the `s2`.
#### Approach
* to decrease cost of comparing arrays, we can also use hashing
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$
 
# 3.02.2023
[6. Zigzag Conversion](https://leetcode.com/problems/zigzag-conversion/description/) medium

[blog post](https://leetcode.com/problems/zigzag-conversion/solutions/3135114/kotlin-simulation/)

```kotlin 
    fun convert(s: String, numRows: Int): String {
        if (numRows <= 1) return s
        // nr = 5
        //
        // 0    8       16        24
        // 1   7 9     15 17     23 25
        // 2  6  10   14   18   22   26   30
        // 3 5    11 13     19 21     27 29
        // 4       12        20        28
        //
        val indices = Array(numRows) { mutableListOf<Int>() }
        var y = 0
        var dy = 1
        for (i in 0..s.lastIndex) {
            indices[y].add(i)
            if (i > 0 && (i % (numRows - 1)) == 0) dy = -dy
            y += dy
        }
        return StringBuilder().apply {
            indices.forEach { it.forEach { append(s[it]) } }
        }.toString()
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/107
#### Intuition

```

        // nr = 5
        //
        // 0    8       16        24
        // 1   7 9     15 17     23 25
        // 2  6  10   14   18   22   26   30
        // 3 5    11 13     19 21     27 29
        // 4       12        20        28
        //

```

We can just simulate zigzag.
#### Approach
Store simulation result in a `[rowsNum][simulation indice]` - matrix, then build the result.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

# 2.02.2023
[953. Verifying an Alien Dictionary](https://leetcode.com/problems/verifying-an-alien-dictionary/description/) easy

[blog post](https://leetcode.com/problems/verifying-an-alien-dictionary/solutions/3130516/kotlin-translate-and-sort/)

```kotlin 
    fun isAlienSorted(words: Array<String>, order: String): Boolean {
        val orderChars = Array<Char>(26) { 'a' }
        for (i in 0..25) orderChars[order[i].toInt() - 'a'.toInt()] = (i + 'a'.toInt()).toChar()
        val arr = Array<String>(words.size) { 
            words[it].map { orderChars[it.toInt() - 'a'.toInt()] }.joinToString("")
        }
        
        val sorted = arr.sorted()
        for (i in 0..arr.lastIndex) if (arr[i] != sorted[i]) return false
        return true
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/106
#### Intuition
For the example `hello` and order `hlabcdefgijkmnopqrstuvwxyz` we must translate like this: `h` -> `a`, `l` -> `b`, `a` -> `c` and so on. Then we can just use `compareTo` to check the order.
#### Approach
Just translate and then sort and compare. (But we can also just scan linearly and compare).
#### Complexity
- Time complexity:
  $$O(n\log_2{n})$$
- Space complexity:
  $$O(n)$$
 
# 1.02.2023
[1071. Greatest Common Divisor of Strings](https://leetcode.com/problems/greatest-common-divisor-of-strings/description/) easy

[blog post](https://leetcode.com/problems/greatest-common-divisor-of-strings/solutions/3125925/kotlin-gcd/)

```kotlin 
    fun gcdOfStrings(str1: String, str2: String): String {
        if (str1 == "" || str2 == "") return ""
        if (str1.length == str2.length) return if (str1 == str2) str1 else ""
        fun gcd(a: Int, b: Int): Int {
            return if (a == 0) b
            else gcd(b % a, a)
        }
        val len = gcd(str1.length, str2.length)
        for (i in 0..str1.lastIndex)  if (str1[i] != str1[i % len]) return ""
        for (i in 0..str2.lastIndex)  if (str2[i] != str1[i % len]) return ""
        return str1.substring(0, len)
        
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/105
#### Intuition
Consider the following example: `ababab` and `abab`. 
If we scan them linearly, we see, the common part is `abab`. 
Now, we need to check if the last part from the first `abab_ab` is a part of the common part: `ab` vs `abab`. 
This can be done recursively, and we come to the final consideration: `"" vs "ab"`. 
That all procedure give us the common divisor - `ab`.
The actual hint is in the method's name ;)

#### Approach
We can first find the length of the greatest common divisor, then just check both strings.

#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(n)$$

# 31.01.2023
[1626. Best Team With No Conflicts](https://leetcode.com/problems/best-team-with-no-conflicts/description/) medium

[blog post](https://leetcode.com/problems/best-team-with-no-conflicts/solutions/3123505/kotlin-dfs-memo/)

```kotlin 
    fun bestTeamScore(scores: IntArray, ages: IntArray): Int {
        val dp = Array(scores.size + 1) { IntArray(1001) { -1 }}
        val indices = scores.indices.toMutableList()
        indices.sortWith(compareBy( { scores[it] }, { ages[it] } ))
        fun dfs(curr: Int, prevAge: Int): Int {
            if (curr == scores.size) return 0
            if (dp[curr][prevAge] != -1) return dp[curr][prevAge]
            val ind = indices[curr]
            val age = ages[ind]
            val score = scores[ind]
            val res = maxOf(
                dfs(curr + 1, prevAge),
                if (age < prevAge) 0  else score + dfs(curr + 1, age)
            )
            dp[curr][prevAge] = res
            return res
        }
        return dfs(0, 0)
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/103
#### Intuition
If we sort arrays by `score` and `age`, then every next item will be with  `score` bigger than previous. 
If current `age` is less than previous, then we can't take it, as `score` for current `age` can't be bigger than previous. 
Let's define `dp[i][j]` is a maximum score for a team in `i..n` sorted slice, and `j` is a maximum age for that team.
#### Approach
We can use DFS to search all the possible teams and memorize the result in dp cache.
#### Complexity
- Time complexity:
  $$O(n^2)$$, we can only visit n by n combinations of pos and age
- Space complexity:
  $$O(n^2)$$

# 30.01.2023
[1137. N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/description/) easy

[blog post](https://leetcode.com/problems/n-th-tribonacci-number/solutions/3116945/kotlin-code-golf/)

```kotlin 
    fun tribonacci(n: Int): Int = if (n < 2) n else {
        var t0 = 0
        var t1 = 1
        var t2 = 1
        repeat(n - 2) {
            t2 += (t0 + t1).also { 
                t0 = t1
                t1 = t2
            }
        }
        t2
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/102
#### Intuition
Just do what is asked.
#### Approach
* another way is to use dp cache
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$
 
# 29.01.2023
[460. LFU Cache](https://leetcode.com/problems/lfu-cache/description/) hard

[blog post](https://leetcode.com/problems/lfu-cache/solutions/3112799/kotlin-treemap-linkedhashset-o-log-sqrt-n)

```kotlin 
class LFUCache(val capacity: Int) {
    data class V(val key: Int, val value: Int, val freq: Int)
    val mapKV = mutableMapOf<Int, V>()
    val freqToAccessListOfK = TreeMap<Int, LinkedHashSet<V>>()

    fun get(key: Int): Int {
        val v = mapKV.remove(key)
        if (v == null) return -1
        increaseFreq(v, v.value)
        return v.value
    }

    fun getAccessListForFreq(freq: Int) = freqToAccessListOfK.getOrPut(freq, { LinkedHashSet<V>() })

    fun increaseFreq(v: V, value: Int) {
        val oldFreq = v.freq
        val newFreq = oldFreq + 1
        val newV = V(v.key, value, newFreq)
        mapKV[v.key] = newV
        val accessList = getAccessListForFreq(oldFreq)
        accessList.remove(v)
        if (accessList.isEmpty()) freqToAccessListOfK.remove(oldFreq)
        getAccessListForFreq(newFreq).add(newV)
    }

    fun put(key: Int, value: Int) {
        if (capacity == 0) return
        val oldV = mapKV[key]
        if (oldV == null) {
            if (mapKV.size == capacity) {
                val lowestFreq = freqToAccessListOfK.firstKey()
                val accessList = freqToAccessListOfK[lowestFreq]!!
                val iterator = accessList.iterator()
                val leastFreqV = iterator.next()
                iterator.remove()
                mapKV.remove(leastFreqV.key)
                if (accessList.isEmpty()) freqToAccessListOfK.remove(lowestFreq)
            }
            val v = V(key, value, 1)
            mapKV[key] = v
            getAccessListForFreq(1).add(v)
        } else {
            increaseFreq(oldV, value)
        }
    }

}

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/101
#### Intuition
Let's store access-time list in a buckets divided by access-count frequencies. We can store each bucked in a `TreeMap`, that will give us O(1) time to get the least frequent list. For the list we can use `LinkedHashSet`, that can give us O(1) operations for `remove`, `removeFirst` and `add` and will help to maintain access order.
#### Approach
* one thing to note, on each `increaseFreq` operation we are retrieving a random item from TreeMap, that increases time to O(log(F)), where F is a unique set of frequencies.
* How many unique access frequencies `k` we can have if there is a total number of `N` operations? If sequence `1,2,3...k-1, k` is our unique set, then `1+2+3+...+(k-1)+k = N`. Or:
  $$
  1+2+3+\cdots+k=\sum_{n=1}^{k}i = k(k-1)/2 = N
  $$
  so,
  $$
  k = \sqrt{N}
  $$
#### Complexity
- Time complexity:
  $$O(\log_2(\sqrt{N}))$$
- Space complexity:
  $$O(\log_2(\sqrt{N}))$$

# 28.01.2023
[352. Data Stream as Disjoint Intervals](https://leetcode.com/problems/data-stream-as-disjoint-intervals/description/) hard

[blog post](https://leetcode.com/problems/data-stream-as-disjoint-intervals/solutions/3108727/kotlin-linked-list/)

```kotlin 
class SummaryRanges() {
    data class Node(var start: Int, var end: Int, var next: Node? = null) 

    val root = Node(-1, -1)

    fun mergeWithNext(n: Node?): Boolean {
        if (n == null) return false
        val curr = n
        val next = n.next
        if (next == null) return false
        val nextNext = next.next
        if (next.start - curr.end <= 1) {
            curr.end = next.end
            curr.next = nextNext
            return true
        }
        return false
    }

    fun addNum(value: Int) {
        var n = root
        while (n.next != null && n.next!!.start < value) n = n.next!!
        if (value in n.start..n.end) return
        n.next = Node(value, value, n.next)
        if (n != root && mergeWithNext(n)) 
            mergeWithNext(n)
        else 
            mergeWithNext(n.next)
    }

    fun getIntervals(): Array<IntArray> {
        val list = mutableListOf<IntArray>()
        var n = root.next
        while (n != null) {
            list.add(intArrayOf(n.start, n.end)) 
            n = n.next
        }
        return list.toTypedArray()
    }

}

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/100
#### Intuition
In Kotlin there is no way around to avoid the O(n) time of an operation while building the result array. 
And there is no way to insert to the middle of the array in a less than O(n) time. 
So, the only way is to use the linked list, and to walk it linearly.

#### Approach
* careful with merge
#### Complexity
- Time complexity:
  $$O(IN)$$, I - number of the intervals
- Space complexity:
  $$O(I)$$

# 27.01.2023
[472. Concatenated Words](https://leetcode.com/problems/concatenated-words/description/) hard

[blog post](https://leetcode.com/problems/concatenated-words/solutions/3104496/kotlin-trie/)

```kotlin 
    data class Trie(val ch: Char = '.', var isWord: Boolean = false) {
        val next = Array<Trie?>(26) { null }
        fun ind(c: Char) = c.toInt() - 'a'.toInt()
        fun exists(c: Char) = next[ind(c)] != null
        operator fun get(c: Char): Trie {
            val ind = ind(c)
            if (next[ind] == null) next[ind] = Trie(c)
            return next[ind]!!
        }
    }
    fun findAllConcatenatedWordsInADict(words: Array<String>): List<String> {
        val trie = Trie()
        words.forEach { word ->
            var t = trie
            word.forEach { t = t[it] }
            t.isWord = true
        }
        val res = mutableListOf<String>()
        words.forEach { word ->
            var tries = ArrayDeque<Pair<Trie,Int>>()
            tries.add(trie to 0)
            for (c in word) {
                repeat(tries.size) {
                    val (t, wc) = tries.poll()
                    if (t.exists(c)) {
                        val curr = t[c]
                        if (curr.isWord)  tries.add(trie to (wc + 1))
                        tries.add(curr to wc)
                    }
                }
            }
            if (tries.any { it.second > 1 && it.first === trie } ) res.add(word)
        }
        return res
    }

```

#### Telegram
https://t.me/leetcode_daily_unstoppable/99
#### Intuition
When we scan a word we must know if current suffix is a word. Trie data structure will help.

#### Approach
* first, scan all the words, and fill the Trie
* next, scan again, and for each suffix begin a new scan from the root of the trie
* preserve a word count for each of the possible suffix concatenation
#### Complexity
- Time complexity:
  $$O(nS)$$, S - is a max suffix count in one word
- Space complexity:
  $$O(n)$$

# 26.01.2023
[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/description/) medium

[https://t.me/leetcode_daily_unstoppable/98](https://t.me/leetcode_daily_unstoppable/98)

[blog post](https://leetcode.com/problems/cheapest-flights-within-k-stops/solutions/3102372/kotlin-bellman-ford/)

```kotlin 
    fun findCheapestPrice(n: Int, flights: Array<IntArray>, src: Int, dst: Int, k: Int): Int {
        var dist = IntArray(n) { Int.MAX_VALUE }
        dist[src] = 0
        repeat(k + 1) {
            val nextDist = dist.clone()
            flights.forEach { (from, to, price) ->
                if (dist[from] != Int.MAX_VALUE && dist[from] + price < nextDist[to]) 
                    nextDist[to] = dist[from] + price
            }
            dist = nextDist
        }
        return if (dist[dst] == Int.MAX_VALUE) -1 else dist[dst]
    }

```

#### Intuition
DFS and Dijkstra gives TLE.
As we need to find not just shortest path price, but only for `k` steps, naive Bellman-Ford didn't work. 
Let's define `dist`, where `dist[i]` - the shortest distance from `src` node to `i`-th node. 
We initialize it with `MAX_VALUE`, and `dist[src]` is 0 by definition. 
Next, we walk exactly `k` steps, on each of them, trying to minimize price. 
If we have known distance to node `a`, `dist[a] != MAX`. 
And if there is a link to node `b` with `price(a,b)`, then we can optimize like this `dist[b] = min(dist[b], dist[a] + price(a,b))`. 
Because we're starting from a single node `dist[0]`, we will increase distance only once per iteration. 
So, making `k` iterations made our path exactly `k` steps long.

#### Approach
* by the problem definition, path length is `k+1`, not just `k`
* we can't optimize a path twice in a single iteration, because then it will overreach to the next step before the current is finished. 
* That's why we only compare distance from the previous step.
 
Space: O(kE), Time: O(k)

# 25.01.2023
[2359. Find Closest Node to Given Two Nodes](https://leetcode.com/problems/find-closest-node-to-given-two-nodes/description/) medium

[https://t.me/leetcode_daily_unstoppable/97](https://t.me/leetcode_daily_unstoppable/97)

[blog post](https://leetcode.com/problems/find-closest-node-to-given-two-nodes/solutions/3096815/kotlin-dfs/)

```kotlin 
    fun closestMeetingNode(edges: IntArray, node1: Int, node2: Int): Int {
        val distances = mutableMapOf<Int, Int>()
        var n = node1
        var dist = 0
        while (n != -1) {
            if (distances.contains(n)) break
            distances[n] = dist
            n = edges[n]
            dist++
        }
        n = node2
        dist = 0
        var min = Int.MAX_VALUE
        var res = -1
        while (n != -1) {
            if (distances.contains(n)) {
                val one = distances[n]!!
                val max = maxOf(one, dist)
                if (max < min || max == min && n < res) {
                    min = max
                    res = n
                }
            }
            val tmp = edges[n]
            edges[n] = -1
            n = tmp
            dist++
        }
        return res
    }

```

![image.png](https://assets.leetcode.com/users/images/b855b06b-ac15-403d-ad0e-13b26850da26_1674632188.3267126.png)

We can walk with DFS and remember all distances, then compare them and choose those with minimum of maximums.
* we can use `visited` set, or modify an input
* corner case: don't forget to also store starting nodes

Space: O(n), Time: O(n)

# 24.01.2023
[909. Snakes and Ladders](https://leetcode.com/problems/snakes-and-ladders/description/) medium

[https://t.me/leetcode_daily_unstoppable/96](https://t.me/leetcode_daily_unstoppable/96)

[blog post](https://leetcode.com/problems/snakes-and-ladders/solutions/3094842/kotlin-bfs/)

```kotlin 
    fun snakesAndLadders(board: Array<IntArray>): Int {
        fun col(pos: Int): Int {
            return if (((pos/board.size) % 2) == 0) 
                    (pos % board.size)
                else 
                    (board.lastIndex - (pos % board.size))
        }
        val last = board.size * board.size
        var steps = 0
        val visited = mutableSetOf<Int>()
        with(ArrayDeque<Int>().apply { add(1) }) {
            while (isNotEmpty() && steps <= last) {
                repeat(size) {
                    var curr = poll()
                    val jump = board[board.lastIndex - (curr-1)/board.size][col(curr-1)]
                    if (jump != -1) curr = jump
                    if (curr == last) return steps
                    for (i in 1..6)  
                        if (visited.add(curr + i) && curr + i <= last) add(curr + i) 
                }
                steps++
            }
        }
        return -1
    }

```

In each step, we can choose the best outcome, so we need to travel all of them in the parallel and calculate steps number. This is a BFS.

We can avoid that strange order by iterating it and store into the linear array. Or just invent a formula for row and column by given index.

Space: O(n^2), Time: O(n^2), n is a grid size

# 23.01.2023
[997. Find the Town Judge](https://leetcode.com/problems/find-the-town-judge/description/) easy

[https://t.me/leetcode_daily_unstoppable/95](https://t.me/leetcode_daily_unstoppable/95)

[blog post](https://leetcode.com/problems/find-the-town-judge/solutions/3089245/kotlin-map-and-set/)

```kotlin 
    fun findJudge(n: Int, trust: Array<IntArray>): Int {
        val judges = mutableMapOf<Int, MutableSet<Int>>()
        for (i in 1..n) judges[i] = mutableSetOf()
        val notJudges = mutableSetOf<Int>()
        trust.forEach { (from, judge) ->
            judges[judge]!! += from
            notJudges += from
        }
        judges.forEach { (judge, people) ->
            if (people.size == n - 1 
                && !people.contains(judge) 
                && !notJudges.contains(judge)) 
                return judge
        }
        return -1
    }

```

We need to count how much trust have each judge and also exclude all judges that have trust in someone.

* use map and set
* there is a better solution with just counting of trust, but it is not that clear to understand and prove
 
Space: O(max(N, T)), Time: O(max(N, T))

# 22.01.2023
[131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/description/) medium

[https://t.me/leetcode_daily_unstoppable/93](https://t.me/leetcode_daily_unstoppable/93)

[blog post](https://leetcode.com/problems/palindrome-partitioning/solutions/3085293/kotlin-dp-and-dfs/)

```kotlin 
    fun partition(s: String): List<List<String>> {
        val dp = Array(s.length) { BooleanArray(s.length) { false } }
        for (from in s.lastIndex downTo 0) 
            for (to in from..s.lastIndex) 
                dp[from][to] = s[from] == s[to] && (from == to || from == to - 1 || dp[from+1][to-1])
        val res = mutableListOf<List<String>>()
        fun dfs(pos: Int, partition: MutableList<String>) {
            if (pos == s.length) res += partition.toList()
            for (i in pos..s.lastIndex) 
                if (dp[pos][i]) {
                    partition += s.substring(pos, i+1)
                    dfs(i+1, partition)
                    partition.removeAt(partition.lastIndex)
                }
        }
        dfs(0, mutableListOf())
        return res
    }

```

First, we need to be able to quickly tell if some range `a..b` is a palindrome. 
Let's `dp[a][b]` indicate that range `a..b` is a palindrome.
Then the following is true: `dp[a][b] = s[a] == s[b] && dp[a+1][b-1]`, also two corner cases, when `a == b` and `a == b-1`. 
For example, "a" and "aa".
* Use `dp` for precomputing palindrome range answers.
* Try all valid partitions with backtracking.
 
Space: O(2^N), Time: O(2^N)

# 21.01.2023
[93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/description/) medium

[https://t.me/leetcode_daily_unstoppable/92](https://t.me/leetcode_daily_unstoppable/92)

[blog post](https://leetcode.com/problems/restore-ip-addresses/solutions/3080471/kotlin-dfs-backtracking/)

```kotlin 
    fun restoreIpAddresses(s: String): List<String> {
	val res = mutableSetOf<String>()
	fun dfs(pos: Int, nums: MutableList<Int>) {
		if (pos == s.length || nums.size > 4) {
			if (nums.size == 4) res += nums.joinToString(".")
			return
		}
		var n = 0

		for (i in pos..s.lastIndex) {
			n = n*10 + s[i].toInt() - '0'.toInt()
			if (n > 255) break
			nums += n
			dfs(i + 1, nums)
			nums.removeAt(nums.lastIndex)
			if (n == 0) break
		}
	}
	dfs(0, mutableListOf())
	return res.toList()
}

```

So, the size of the problem is small. We can do full DFS.
At every step, choose either take a number or split. Add to the solution if the result is good.

* use set for results
* use backtracking to save some space

Some optimizations: 
* exit early when nums.size > 5, 
* use math to build a number instead of parsing substring

Space: O(2^N), Time: O(2^N)

# 20.01.2023
[491. Non-decreasing Subsequences](https://leetcode.com/problems/non-decreasing-subsequences/description/) medium

[https://t.me/leetcode_daily_unstoppable/91](https://t.me/leetcode_daily_unstoppable/91)

[blog post](https://leetcode.com/problems/non-decreasing-subsequences/solutions/3075577/kotlin-backtraking-set/)

```kotlin 
    fun findSubsequences(nums: IntArray): List<List<Int>> {
        val res = mutableSetOf<List<Int>>()
        fun dfs(pos: Int, currList: MutableList<Int>) {
            if (currList.size > 1) res += currList.toList()
            if (pos == nums.size) return
            val currNum = nums[pos]
            //not add
            dfs(pos + 1, currList)
            //to add
            if (currList.isEmpty() || currList.last()!! <= currNum) {
                currList += currNum
                dfs(pos + 1, currList)
                currList.removeAt(currList.lastIndex)
            }
        }
        dfs(0, mutableListOf())
        return res.toList()
    }

```

Notice the size of the problem, we can do a brute force search for all solutions. Also, we only need to store the unique results, so we can store them in a set.

* we can reuse pre-filled list and do backtracking on the return from the DFS.
 
Space: O(2^N) to store the result, Time: O(2^N) for each value we have two choices, and we can build a binary tree of choices with the 2^n number of elements. 
 
# 19.01.2023
[974. Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/) medium

[https://t.me/leetcode_daily_unstoppable/90](https://t.me/leetcode_daily_unstoppable/90)

[blog post](https://leetcode.com/problems/subarray-sums-divisible-by-k/solutions/3073473/kotlin-prefix-sum-and-remainders/)

```kotlin 
    fun subarraysDivByK(nums: IntArray, k: Int): Int {
        // 4 5 0 -2 -3 1    k=5   count
        // 4                4:1   0
        //   9              4:2   +1
        //     9            4:3   +2
        //       7          2:1   
        //          4       4:4   +3
        //             5    0:2   +1
        // 2 -2 2 -4       k=6
        // 2               2:1
        //    0            0:2    +1
        //      2          2:2    +1
        //        -2       2:3    +2
        // 1 2 13 -2 3  k=7
        // 1
        //   3
        //     16
        //        14
        //          17 (17-1*7= 10, 17-2*7=3, 17-3*7=-4, 17-4*7 = -11)
        val freq = mutableMapOf<Int, Int>()
        freq[0] = 1
        var sum = 0 
        var res = 0
        nums.forEach {
            sum += it
            var ind = (sum % k)
            if (ind < 0) ind += k
            val currFreq = freq[ind] ?: 0
            res += currFreq
            freq[ind] = 1 + currFreq
        }
        return res
    }

```

We need to calculate a running sum. 
For every current sum, we need to find any subsumes that are divisible by k, so `sum[i]: (sum[i] - sum[any prev]) % k == 0`. 
Or, `sum[i] % k == sum[any prev] % k`. 
Now, we need to store all `sum[i] % k` values, count them and add to result.

We can save frequency in a map, or in an array [0..k], because all the values are from that range.

Space: O(N), Time: O(N)

# 18.01.2023
[918. Maximum Sum Circular Subarray](https://leetcode.com/problems/maximum-sum-circular-subarray/description/) medium

[https://t.me/leetcode_daily_unstoppable/89](https://t.me/leetcode_daily_unstoppable/89)

[blog post](https://leetcode.com/problems/maximum-sum-circular-subarray/solutions/3069120/kotlin-invert-the-problem/)

```kotlin 
    fun maxSubarraySumCircular(nums: IntArray): Int {
        var maxEndingHere = 0
        var maxEndingHereNegative = 0
        var maxSoFar = Int.MIN_VALUE
        var total = nums.sum()
        nums.forEach {
            maxEndingHere += it
            maxEndingHereNegative += -it
            maxSoFar = maxOf(maxSoFar, maxEndingHere, if (total == -maxEndingHereNegative) Int.MIN_VALUE else total+maxEndingHereNegative)
            if (maxEndingHere < 0) {
                maxEndingHere = 0
            }
            if (maxEndingHereNegative < 0) {
                maxEndingHereNegative = 0
            }
        }
        return maxSoFar
    }

```

Simple Kadane's Algorithm didn't work when we need to keep a window of particular size. 
One idea is to invert the problem and find the minimum sum and subtract it from the total.

One corner case:
* we can't subtract all the elements when checking the negative sum.

Space: O(1), Time: O(N)

# 17.01.2023
[926. Flip String to Monotone Increasing](https://leetcode.com/problems/flip-string-to-monotone-increasing/description/) medium

[https://t.me/leetcode_daily_unstoppable/88](https://t.me/leetcode_daily_unstoppable/88)

[blog post](https://leetcode.com/problems/flip-string-to-monotone-increasing/solutions/3062530/kotlin-dp/)

```kotlin 
    fun minFlipsMonoIncr(s: String): Int {
        // 010110  dp0  dp1    min
        // 0       0    0      0
        //  1      1    0      1
        //   0     1    1      1
        //    1    2    1      1
        //     1   3    1      1
        //      0  3    2      2
        var dp0 = 0
        var dp1 = 0

        for (i in 0..s.lastIndex) {
            dp0 = if (s[i] == '0') dp0 else 1 + dp0
            dp1 = if (s[i] == '1') dp1 else 1 + dp1
            if (dp0 <= dp1) dp1 = dp0
        }
        
        return minOf(dp0, dp1)
    }

```

We can propose the following rule: let's define `dp0[i]` is a min count of flips from `1` to `0` in the `0..i` interval. 
Let's also define `dp1[i]` is a min count of flips from `0` to `1` in the `0..i` interval. 
We observe that `dp0[i] = dp0[i-1] + (flip one to zero? 1 : 0)` and `dp1[i] = dp1[i-1] + (flip zero to one? 1 : 0)`. 
One special case: if on the interval `0..i` one-to-zero flips count is less than zero-to-one then we prefer to flip everything to zeros, and `dp1[i]` in that case becomes `dp0[i]`.

Just write down what is described above.
* dp arrays can be simplified to single variables.

Space: O(1), Time: O(N)

# 16.01.2023
[57. Insert Interval](https://leetcode.com/problems/insert-interval/description/) medium

[https://t.me/leetcode_daily_unstoppable/87](https://t.me/leetcode_daily_unstoppable/87)

[blog post](https://leetcode.com/problems/insert-interval/solutions/3057540/kotlin-one-pass/)

```kotlin 
    fun insert(intervals: Array<IntArray>, newInterval: IntArray): Array<IntArray> {
        val res = mutableListOf<IntArray>()
        var added = false
        fun add() {
            if (!added) {
                added = true
                if (res.isNotEmpty() && res.last()[1] >= newInterval[0]) {
                    res.last()[1] = maxOf(res.last()[1], newInterval[1])
                } else res += newInterval
            }
        }
        intervals.forEach { interval -> 
            if (newInterval[0] <= interval[0]) add()
            
            if (res.isNotEmpty() && res.last()[1] >= interval[0]) {
                res.last()[1] = maxOf(res.last()[1], interval[1])
            } else  res += interval
        }
        add()
       
        return res.toTypedArray()
    }

```

There is no magic, just be careful with corner cases.

Make another list, and iterate interval, merging them and adding at the same time.
* don't forget to add `newInterval` if it is not added after iteration.

Space: O(N), Time: O(N)

# 15.01.2023
[2421. Number of Good Paths](https://leetcode.com/problems/number-of-good-paths/) hard

[https://t.me/leetcode_daily_unstoppable/86](https://t.me/leetcode_daily_unstoppable/86)

[blog post](https://leetcode.com/problems/number-of-good-paths/solutions/3054534/kotlin-union-find-was-hard/)

```kotlin 
    fun numberOfGoodPaths(vals: IntArray, edges: Array<IntArray>): Int {
        if (edges.size == 0) return vals.size
        edges.sortWith(compareBy(  { maxOf( vals[it[0]], vals[it[1]] ) }  ))
        val uf = IntArray(vals.size) { it }
        val freq = Array(vals.size) { mutableMapOf(vals[it] to 1) }
        fun find(x: Int): Int {
            var p = x
            while (uf[p] != p) p = uf[p]
            uf[x] = p
            return p
        }
        fun union(a: Int, b: Int): Int {
            val rootA = find(a)
            val rootB = find(b)
            if (rootA == rootB) return 0
            uf[rootA] = rootB
            val vMax = maxOf(vals[a], vals[b]) // if we connect tree [1-3] to tree [2-1], only `3` matters
            val countA = freq[rootA][vMax] ?:0
            val countB = freq[rootB][vMax] ?:0
            freq[rootB][vMax] = countA + countB
            return countA * countB
        }
        return edges.map { union(it[0], it[1])}.sum()!! + vals.size
    }

```

The naive solution with single DFS and merging frequency maps gives TLE. 
Now, use hint, and they tell you to sort edges and use Union-Find :) 
The idea is to connect subtrees, but walk them from smallest to the largest of value. 
When we connect two subtrees, we look at the maximum of each subtree. 
The minimum values don't matter because the path will break at the maximums by definition of the problem.

Use IntArray for Union-Find, and also keep frequencies maps for each root.

Space: O(NlogN), Time: O(N)

# 14.01.2023
[1061. Lexicographically Smallest Equivalent String](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/description/) medium

[https://t.me/leetcode_daily_unstoppable/85](https://t.me/leetcode_daily_unstoppable/85)

[blog post](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/solutions/3049304/kotlin-uniton-find/)

```kotlin 
    fun smallestEquivalentString(s1: String, s2: String, baseStr: String): String {
        val uf = IntArray(27) { it }
        fun find(ca: Char): Int {
            val a = ca.toInt() - 'a'.toInt()
            var x = a
            while (uf[x] != x) x = uf[x]
            uf[a] = x
            return x
        }
        fun union(a: Char, b: Char) {
            val rootA = find(a)
            val rootB = find(b)
            if (rootA != rootB) {
                val max = maxOf(rootA, rootB)
                val min = minOf(rootA, rootB)
                uf[max] = min
            }
        }
        for (i in 0..s1.lastIndex) union(s1[i], s2[i])
        return baseStr.map { (find(it) + 'a'.toInt()).toChar() }.joinToString("")
    }

```

We need to find connected groups, the best way is to use the Union-Find.

Iterate over strings and connect each of their chars.
* to find a minimum, we can select the minimum of the current root.

Space: O(N) for storing a result, Time: O(N)

# 13.01.2023
[2246. Longest Path With Different Adjacent Characters](https://leetcode.com/problems/longest-path-with-different-adjacent-characters/description/) hard

[https://t.me/leetcode_daily_unstoppable/84](https://t.me/leetcode_daily_unstoppable/84)

[blog post](https://leetcode.com/problems/longest-path-with-different-adjacent-characters/solutions/3046179/kotlin-build-graph-dfs/)

```kotlin 
    fun longestPath(parent: IntArray, s: String): Int {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        for (i in 1..parent.lastIndex) 
            if (s[i] != s[parent[i]]) graph.getOrPut(parent[i], { mutableListOf() }) += i
        
        var maxLen = 0
        fun dfs(curr: Int): Int {
            parent[curr] = curr
            var max1 = 0
            var max2 = 0
            graph[curr]?.forEach { 
                val childLen = dfs(it) 
                if (childLen > max1) {
                    max2 = max1
                    max1 = childLen
                } else if (childLen > max2) max2 = childLen
            }
            val childChainLen = 1 + (max1 + max2)
            val childMax = 1 + max1
            maxLen = maxOf(maxLen, childMax, childChainLen)
            return childMax
        }
        for (i in 0..parent.lastIndex) if (parent[i] != i) dfs(i)

        return maxLen
    }

```

Longest path is a maximum sum of the two longest paths of the current node.

Let's build a graph and then recursively iterate it by DFS. We need to find two largest results from the children DFS calls.
* make `parent[i] == i` to store a `visited` state

Space: O(N), Time: O(N), in DFS we visit each node only once.

# 12.01.2023
[1519. Number of Nodes in the Sub-Tree With the Same Label](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label/description/) medium

[https://t.me/leetcode_daily_unstoppable/83](https://t.me/leetcode_daily_unstoppable/83)

[blog post](https://leetcode.com/problems/number-of-nodes-in-the-sub-tree-with-the-same-label/solutions/3039078/kotlin-build-graph-count-by-dfs/)

```kotlin 
fun countSubTrees(n: Int, edges: Array<IntArray>, labels: String): IntArray {
	val graph = mutableMapOf<Int, MutableList<Int>>()
	edges.forEach { (from, to) ->
		graph.getOrPut(from, { mutableListOf() }) += to
		graph.getOrPut(to, { mutableListOf() }) += from
	}
	val answer = IntArray(n) { 0 }
	fun dfs(node: Int, parent: Int, counts: IntArray) {
		val index = labels[node].toInt() - 'a'.toInt()
		val countParents = counts[index]
		counts[index]++
		graph[node]?.forEach {
			if (it != parent) {
				dfs(it, node, counts)
			}
		}
		answer[node] = counts[index] - countParents
	}
	dfs(0, 0, IntArray(27) { 0 })
	return answer
}

```

First, we need to build a graph. Next, just do DFS and count all `'a'..'z'` frequencies in the current subtree.

For building a graph let's use a map, and for DFS let's use a recursion.
* use `parent` node instead of the visited set
* use in-place counting and subtract `count before`

Space: O(N), Time: O(N)

# 11.01.2023
[1443. Minimum Time to Collect All Apples in a Tree](https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/description/) medium

[https://t.me/leetcode_daily_unstoppable/82](https://t.me/leetcode_daily_unstoppable/82)

[blog post](https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/solutions/3036411/kotlin-build-tree-and-count-paths-to-parents/)

```kotlin 
    fun minTime(n: Int, edges: Array<IntArray>, hasApple: List<Boolean>): Int {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) ->
            graph.getOrPut(to, { mutableListOf() }) += from
            graph.getOrPut(from, { mutableListOf() }) += to
        }

        val queue = ArrayDeque<Int>()
        queue.add(0)
        val parents = IntArray(n+1) { it }
        while (queue.isNotEmpty()) {
            val node = queue.poll()
            graph[node]?.forEach {
                if (parents[it] == it && it != 0) {
                    parents[it] = node
                    queue.add(it)
                }
            }
        }
        var time = 0
        hasApple.forEachIndexed { i, has ->
            if (has) {
                var node = i
                while (node != parents[node]) {
                    val parent = parents[node]
                    parents[node] = node
                    node = parent
                    time++
                }
            }
        }
        return time * 2
    }

```

We need to count all paths from apples to 0-node and don't count already walked path.
* notice, that problem definition doesn't state the order of the edges in `edges` array. We need to build the tree first.

First, build the tree, let it be a `parents` array, where `parent[i]` is a parent of the `i`. 
Walk graph with DFS and write the parents. 
Next, walk `hasApple` list and for each apple count parents until reach node `0` or already visited node. 
To mark a node as visited, make it the parent of itself.

Space: O(N), Time: O(N)

# 10.01.2023
[100. Same Tree](https://leetcode.com/problems/same-tree/description/) easy

[https://t.me/leetcode_daily_unstoppable/81](https://t.me/leetcode_daily_unstoppable/81)

[blog post](https://leetcode.com/problems/same-tree/solutions/3028835/kotlin-recursive/)

```kotlin 
fun isSameTree(p: TreeNode?, q: TreeNode?): Boolean =  p == null && q == null || 
            p?.`val` == q?.`val` && isSameTree(p?.left, q?.left) && isSameTree(p?.right, q?.right)

```

Check for the current node and repeat for the children.
Let's write one-liner

Space: O(logN) for stack, Time: O(n)

# 9.01.2023
[144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/description/) easy

[https://t.me/leetcode_daily_unstoppable/80](https://t.me/leetcode_daily_unstoppable/80)

[blog post](https://leetcode.com/problems/binary-tree-preorder-traversal/solutions/3023310/kotlin-morris-stack-recursive/)

```kotlin 
class Solution {
    fun preorderTraversal(root: TreeNode?): List<Int> {
        val res = mutableListOf<Int>()
        var node = root
        while(node != null) {
            res.add(node.`val`)
            if (node.left != null) {
                if (node.right != null) {
                    var rightmost = node.left!!
                    while (rightmost.right != null) rightmost = rightmost.right
                    rightmost.right = node.right
                }
                node = node.left
            } else if (node.right != null) node = node.right
            else node = null
        }
        return res
    }
    fun preorderTraversalStack(root: TreeNode?): List<Int> {
        val res = mutableListOf<Int>()
        var node = root
        val rightStack = ArrayDeque<TreeNode>()
        while(node != null) {
            res.add(node.`val`)
            if (node.left != null) {
                if (node.right != null) {
                    rightStack.addLast(node.right!!) // <-- this step can be replaced with Morris
                    // traversal.
                }
                node = node.left
            } else if (node.right != null) node = node.right
            else if (rightStack.isNotEmpty()) node = rightStack.removeLast()
            else node = null
        }
        return res
    }
    fun preorderTraversalRec(root: TreeNode?): List<Int> = mutableListOf<Int>().apply {
        root?.let {
            add(it.`val`)
            addAll(preorderTraversal(it.left))
            addAll(preorderTraversal(it.right))
        }
    }
        
}

```

Recursive solution is a trivial. For stack solution, we need to remember each `right` node. Morris' solution use the tree modification to save each `right` node in the rightmost end of the left subtree.
Let's implement them all.

Space: O(logN) for stack, O(1) for Morris', Time: O(n)

# 8.01.2023
[149. Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/) hard

[https://t.me/leetcode_daily_unstoppable/79](https://t.me/leetcode_daily_unstoppable/79)

[blog post](https://leetcode.com/problems/max-points-on-a-line/solutions/3018971/kotlin-linear-algebra-n-2/)

```kotlin 
    fun maxPoints(points: Array<IntArray>): Int {
        if (points.size == 1) return 1
        val pointsByTan = mutableMapOf<Pair<Double, Double>, HashSet<Int>>()
        fun gcd(a: Int, b: Int): Int {
            return if (b == 0) a else gcd(b, a%b)
        }
        for (p1Ind in points.indices) {
            val p1 = points[p1Ind]
            for (p2Ind in (p1Ind+1)..points.lastIndex) {
                val p2 = points[p2Ind]
                val x1 = p1[0]
                val x2 = p2[0]
                val y1 = p1[1]
                val y2 = p2[1]
                var dy = y2 - y1
                var dx = x2 - x1
                val greatestCommonDivider = gcd(dx, dy)
                dy /= greatestCommonDivider
                dx /= greatestCommonDivider
                val tan = dy/dx.toDouble()
                val b = if (dx == 0) x1.toDouble() else (x2*y1 - x1*y2 )/(x2-x1).toDouble()
                val line = pointsByTan.getOrPut(tan to b, { HashSet() })
                line.add(p1Ind)
                line.add(p2Ind)
            }
        }
        return pointsByTan.values.maxBy { it.size }?.size?:0
    }

```

Just do the linear algebra to find all the lines through each pair of points.
Store `slope` and `b` coeff in the hashmap. Also, compute `gcd` to find precise slope. In this case it works for `double` precision slope, but for bigger numbers we need to store `dy` and `dx` separately in `Int` precision.

Space: O(n^2), Time: O(n^2)

# 7.01.2023
[134. Gas Station](https://leetcode.com/problems/gas-station/description/) medium

[https://t.me/leetcode_daily_unstoppable/78](https://t.me/leetcode_daily_unstoppable/78)

[blog post](https://leetcode.com/problems/gas-station/solutions/3013707/kotlin-greedy/)

```kotlin 
    fun canCompleteCircuit(gas: IntArray, cost: IntArray): Int {
        var sum = 0
        var minSum = gas[0]
        var ind = -1
        for (i in 0..gas.lastIndex) {
            sum += gas[i] - cost[i]
            if (sum < minSum) {
                minSum = sum
                ind = (i+1) % gas.size
            }
        }
        return if (sum < 0) -1 else ind
    }

```

We can start after the station with the minimum `decrease` in gasoline.
![image.png](https://assets.leetcode.com/users/images/252d5b9e-b28b-4306-95bc-b37c1afed1b9_1673095767.9064982.png)
Calculate running gasoline volume and find the minimum of it. If the total net gasoline is negative, there is no answer.

Space: O(1), Time: O(N)

# 6.01.2023
[1833. Maximum Ice Cream Bars](https://leetcode.com/problems/maximum-ice-cream-bars/description/) medium

[https://t.me/leetcode_daily_unstoppable/77](https://t.me/leetcode_daily_unstoppable/77)

[blog post](https://leetcode.com/problems/maximum-ice-cream-bars/solutions/3007210/kotlin-greedy/)

```kotlin 
    fun maxIceCream(costs: IntArray, coins: Int): Int {
       costs.sort() 
       var coinsRemain = coins
       var iceCreamCount = 0
       for (i in 0..costs.lastIndex) {
           coinsRemain -= costs[i]
           if (coinsRemain < 0) break
           iceCreamCount++
       }
       return iceCreamCount
    }

```

The `maximum ice creams` would be if we take as many `minimum costs` as possible
Sort the `costs` array, then greedily iterate it and buy ice creams until all the coins are spent.

Space: O(1), Time: O(NlogN) (there is also O(N) solution based on count sort)

# 5.01.2023
[452. Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/) medium

[https://t.me/leetcode_daily_unstoppable/75](https://t.me/leetcode_daily_unstoppable/75)

[blog post](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/solutions/3002258/kotlin-sort-line-sweep/)

```kotlin 
    fun findMinArrowShots(points: Array<IntArray>): Int {
        if (points.isEmpty()) return 0
        if (points.size == 1) return 1
        Arrays.sort(points, Comparator<IntArray> { a, b -> 
            if (a[0] == b[0]) a[1].compareTo(b[1]) else a[0].compareTo(b[0]) })
        var arrows = 1
        var arrX = points[0][0]
        var minEnd = points[0][1]
        for (i in 1..points.lastIndex) {
            val (start, end) = points[i]
            if (minEnd < start) {
                arrows++
                minEnd = end
            }
            if (end < minEnd) minEnd = end
            arrX = start
        }
        return arrows
    }

```

The optimal strategy to achieve the minimum number of arrows is to find the maximum overlapping intervals. For this task, we can sort the points by their `start` and `end` coordinates and use line sweep technique. Overlapping intervals are separate if their `minEnd` is less than `start` of the next interval. `minEnd` - the minimum of the `end`'s of the overlapping intervals.
Let's move the arrow to each `start` interval and fire a new arrow if this `start` is greater than `minEnd`.
* for sorting without Int overflowing, use `compareTo` instead of subtraction
* initial conditions are better to initialize with the first interval and iterate starting from the second

Space: O(1), Time: O(NlogN)

# 4.01.2023
[2244. Minimum Rounds to Complete All Tasks](https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/description/) medium

[https://t.me/leetcode_daily_unstoppable/74](https://t.me/leetcode_daily_unstoppable/74)

[blog post](https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/solutions/2997205/kotlin-dfs-memo/)

```kotlin 
    fun minimumRounds(tasks: IntArray): Int {
        val counts = mutableMapOf<Int, Int>()
        tasks.forEach { counts[it] = 1 + counts.getOrDefault(it, 0)}
        var round = 0
        val cache = mutableMapOf<Int, Int>()
        fun fromCount(count: Int): Int {
            if (count == 0) return 0
            if (count < 0 || count == 1) return -1
            return if (count % 3 == 0) {
                count/3
            } else {
                cache.getOrPut(count, {
                    var v = fromCount(count - 3)
                    if (v == -1) v = fromCount(count - 2)
                    if (v == -1) -1 else 1 + v
                })
            }
        }
        counts.values.forEach { 
            val rounds = fromCount(it)
            if (rounds == -1) return -1
            round += rounds
        }
        return round
    }

```

For the optimal solution, we must take as many 3's of tasks as possible, then take 2's in any order.
First, we need to count how many tasks of each type there are. Next, we need to calculate the optimal `rounds` for the current tasks type count. There is a math solution, but ultimately we just can do DFS

Space: O(N), Time: O(N), counts range is always less than N

# 3.01.2023
[944. Delete Columns to Make Sorted](https://leetcode.com/problems/delete-columns-to-make-sorted/description/) easy

[https://t.me/leetcode_daily_unstoppable/73](https://t.me/leetcode_daily_unstoppable/73)

[blog post](https://leetcode.com/problems/delete-columns-to-make-sorted/solutions/2992229/kotlin-do-what-is-asked/)

```kotlin 
    fun minDeletionSize(strs: Array<String>): Int =
       (0..strs[0].lastIndex).asSequence().count { col ->
           (1..strs.lastIndex).asSequence().any { strs[it][col] < strs[it-1][col] }
        } 

```

Just do what is asked.
We can use Kotlin's `sequence` api.

Space: O(1), Time: O(wN)

# 2.01.2023
[520. Detect Capital](https://leetcode.com/problems/detect-capital/description/) easy

[https://t.me/leetcode_daily_unstoppable/72](https://t.me/leetcode_daily_unstoppable/72)

[blog post](https://leetcode.com/problems/detect-capital/solutions/2985088/kotlin-as-is/)

```kotlin 
    fun detectCapitalUse(word: String): Boolean =
       word.all { Character.isUpperCase(it) } ||
       word.all { Character.isLowerCase(it) } ||
       Character.isUpperCase(word[0]) && word.drop(1).all { Character.isLowerCase(it) }

```

We can do this optimally by checking the first character and then checking all the other characters in a single pass. Or we can write a more understandable code that directly translates from the problem description.
Let's write one-liner.

Space: O(1), Time: O(N)

# 1.01.2023
[290. Word Pattern](https://leetcode.com/problems/word-pattern/description/) easy

[https://t.me/leetcode_daily_unstoppable/71](https://t.me/leetcode_daily_unstoppable/71)

[blog post](https://leetcode.com/problems/word-pattern/solutions/2978765/kotlin-just-do/)

```kotlin 
    fun wordPattern(pattern: String, s: String): Boolean {
        val charToWord = Array<String>(27) { "" }
        val words = s.split(" ")
        if (words.size != pattern.length) return false
        words.forEachIndexed { i, w ->
            val cInd = pattern[i].toInt() - 'a'.toInt()

            if (charToWord[cInd] == "") {
                charToWord[cInd] = w
            } else if (charToWord[cInd] != w) return false
        }
        charToWord.sort()
        for (i in 1..26) 
            if (charToWord[i] != "" && charToWord[i] == charToWord[i-1]) 
                return false
        return true
    }

```

Each word must be in 1 to 1 relation with each character in the pattern. We can check this rule.

Use `string[27]` array for `char -> word` relation and also check each char have a unique word assigned.
* don't forget to check lengths

Space: O(N), Time: O(N)

# 31.12.2022
[980. Unique Paths III](https://leetcode.com/problems/unique-paths-iii/description/) hard

[https://t.me/leetcode_daily_unstoppable/69](https://t.me/leetcode_daily_unstoppable/69)

[blog post](https://leetcode.com/problems/unique-paths-iii/solutions/2974827/kotlin-dfs-backtracking/)

```kotlin 
    fun uniquePathsIII(grid: Array<IntArray>): Int {
        var countEmpty = 1
        var startY = 0
        var startX = 0
        for (y in 0..grid.lastIndex) {
            for (x in 0..grid[0].lastIndex) {
                when(grid[y][x]) {
                    0 -> countEmpty++
                    1 -> { startY = y; startX = x}
                    else -> Unit
                }
            }
        }
        fun dfs(y: Int, x: Int): Int {
            if (y < 0 || x < 0 || y >= grid.size || x >= grid[0].size) return 0
            val curr = grid[y][x]
            if (curr == 2) return if (countEmpty == 0) 1 else 0
            if (curr == -1) return 0
            grid[y][x] = -1
            countEmpty--
            val res =  dfs(y-1, x) + dfs(y, x-1) + dfs(y+1, x) + dfs(y, x+1)
            countEmpty++
            grid[y][x] = curr
            return res
        }
        return dfs(startY, startX)
    }

```

There is only `20x20` cells, we can brute-force the solution.
We can use DFS, and count how many empty cells passed. To avoid visiting cells twice, modify `grid` cell and then modify it back, like backtracking.

Space: O(1), Time: O(4^N)

# 30.12.2022
[797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/description/) medium

[https://t.me/leetcode_daily_unstoppable/68](https://t.me/leetcode_daily_unstoppable/68)

[blog post](https://leetcode.com/problems/all-paths-from-source-to-target/solutions/1600383/kotlin-dfs-backtracking-java-iterative-dfs-stack/)

```kotlin 
    fun allPathsSourceTarget(graph: Array<IntArray>): List<List<Int>> {
        val res = mutableListOf<List<Int>>()
        val currPath = mutableListOf<Int>()
        fun dfs(curr: Int) {
            currPath += curr
            if (curr == graph.lastIndex) res += currPath.toList()
            graph[curr].forEach { dfs(it) }
            currPath.removeAt(currPath.lastIndex)
        }
        dfs(0)
        return res
    }

```

We must find all the paths, so there is no shortcuts to the visiting all of them.
One technique is backtracking - reuse existing visited list of nodes.

Space: O(VE), Time: O(VE)

# 29.12.2022
[1834. Single-Threaded CPU](https://leetcode.com/problems/single-threaded-cpu/description/) medium

[https://t.me/leetcode_daily_unstoppable/67](https://t.me/leetcode_daily_unstoppable/67)

[blog post](https://leetcode.com/problems/single-threaded-cpu/solutions/2966855/kotlin-two-heaps/)

```kotlin 
    fun getOrder(tasks: Array<IntArray>): IntArray {
        val pqSource = PriorityQueue<Int>(compareBy(
            { tasks[it][0] },
            { tasks[it][1] },
            { it }
        ))
        (0..tasks.lastIndex).forEach { pqSource.add(it) }
        val pq = PriorityQueue<Int>(compareBy(
            { tasks[it][1] },
            { it }
        ))
        val res = IntArray(tasks.size) { 0 }
        var time = 1 
        for(resPos in 0..tasks.lastIndex) {
            while (pqSource.isNotEmpty() && tasks[pqSource.peek()][0] <= time) {
                pq.add(pqSource.poll())
            }
            if (pq.isEmpty()) {
                //idle
                pq.add(pqSource.poll())
                time = tasks[pq.peek()][0]
            }
            //take task
            val taskInd = pq.poll()
            val task = tasks[taskInd]
            time += task[1]
            res[resPos] = taskInd
        }
        return res
    }

```

First we need to sort tasks by their availability (and other rules), 
then take tasks one by one and add them to another sorted set/heap where their start time doesn't matter, 
but running time and order does. When we take the task from the heap, we increase the time and fill in the heap.
* use two heaps, one for the source of tasks, another for the current available tasks.
* don't forget to increase time to the nearest task if all of them unavailable

Space: O(n), Time: O(nlogn)

# 28.12.2022
[1962. Remove Stones to Minimize the Total](https://leetcode.com/problems/remove-stones-to-minimize-the-total/description/) medium

[https://t.me/leetcode_daily_unstoppable/66](https://t.me/leetcode_daily_unstoppable/66)

[blog post](https://leetcode.com/problems/remove-stones-to-minimize-the-total/solutions/2961725/kotlin-priorityqueue/)

```kotlin 
    fun minStoneSum(piles: IntArray, k: Int): Int {
        val pq = PriorityQueue<Int>() 
        var sum = 0
        piles.forEach { 
            sum += it
            pq.add(-it) 
        }
        for (i in 1..k) {
            if (pq.isEmpty()) break
            val max = -pq.poll()
            if (max == 0) break 
            val newVal = Math.round(max/2.0).toInt()
            sum -= max - newVal
            pq.add(-newVal)
        }
        return sum
    }

```

By the problem definition, intuitively the best strategy is to reduce the maximum each time.
Use `PriorityQueue` to keep track of the maximum value and update it dynamically.
* one can use variable `sum` and update it each time.

Space: O(n), Time: O(nlogn)

# 27.12.2022
[2279. Maximum Bags With Full Capacity of Rocks](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/description/) medium

[https://t.me/leetcode_daily_unstoppable/65](https://t.me/leetcode_daily_unstoppable/65)

[blog post](https://leetcode.com/problems/maximum-bags-with-full-capacity-of-rocks/solutions/2957374/kotlin-sort-greedy/)

```kotlin 
    fun maximumBags(capacity: IntArray, rocks: IntArray, additionalRocks: Int): Int {
       val inds = Array<Int>(capacity.size) { it }
       inds.sortWith(Comparator { a,b -> capacity[a]-rocks[a] - capacity[b] + rocks[b] })
       var rocksRemain = additionalRocks
       var countFull = 0
       for (i in 0..inds.lastIndex) {
           val toAdd = capacity[inds[i]] - rocks[inds[i]]
           if (toAdd > rocksRemain) break
           rocksRemain -= toAdd
           countFull++
       }
       return countFull
    }

```

We can logically deduce that the optimal solution is to take first bags with the smallest empty space.
Make an array of indexes and sort it by difference between `capacity` and `rocks`. Then just simulate rocks addition to each bug from the smallest empty space to the largest.

Space: O(n), Time: O(nlogn)

# 26.12.2022
[55. Jump Game](https://leetcode.com/problems/jump-game/description/) medium

[https://t.me/leetcode_daily_unstoppable/64](https://t.me/leetcode_daily_unstoppable/64)

[blog post](https://leetcode.com/problems/jump-game/solutions/2952687/kotlin-one-pass/)

```kotlin 
    fun canJump(nums: IntArray): Boolean {
       var minInd = nums.lastIndex 
       for (i in nums.lastIndex - 1 downTo 0) {
           if (nums[i] + i >= minInd) minInd = i
       }
       return minInd == 0
    }

```

For any position `i` we can reach the end if there is a `minInd` such that `nums[i] + i >= minInd` and `minInd` is a known to be reaching the end.
We can run from the end and update `minInd` - minimum index reaching the end.

Space: O(1), Time: O(N)

# 25.12.2022
[2389. Longest Subsequence With Limited Sum](https://leetcode.com/problems/longest-subsequence-with-limited-sum/description/) easy

[https://t.me/leetcode_daily_unstoppable/63](https://t.me/leetcode_daily_unstoppable/63)

[blog post](https://leetcode.com/problems/longest-subsequence-with-limited-sum/solutions/2948494/kotlin-sort-prefix-sum-binary-search/)

```kotlin 
    fun answerQueries(nums: IntArray, queries: IntArray): IntArray {
       nums.sort() 
       for (i in 1..nums.lastIndex) nums[i] += nums[i-1]
       return IntArray(queries.size) {
           val ind = nums.binarySearch(queries[it])
           if (ind < 0) -ind-1 else ind+1
       }
    }

```

We can logically deduce that for the maximum number of arguments we need to take as much as possible items from the smallest to the largest.
We can sort items. Then pre-compute `sums[i] = sum from [0..i]`. Then use binary search target sum in sums. Also, can modify `nums` but that's may be not necessary.

Space: O(N), Time: O(NlogN)

# 24.12.2022
[790. Domino and Tromino Tiling](https://leetcode.com/problems/domino-and-tromino-tiling/description/) medium

[https://t.me/leetcode_daily_unstoppable/62](https://t.me/leetcode_daily_unstoppable/62)

[blog post](https://leetcode.com/problems/domino-and-tromino-tiling/solutions/2946811/kotlin-dfs-memo/)

```kotlin 
    fun numTilings(n: Int): Int {
        val cache = Array<Array<Array<Long>>>(n) { Array(2) { Array(2) { -1L }}}
        fun dfs(pos: Int, topFree: Int, bottomFree: Int): Long {
            return when {
                pos > n -> 0L
                pos == n -> if (topFree==1 && bottomFree==1) 1L else 0L
                else -> {
                    var count = cache[pos][topFree][bottomFree]
                    if (count == -1L) {
                        count = 0L
                        when {
                            topFree==1 && bottomFree==1 -> {
                                count += dfs(pos+1, 1, 1) // vertical
                                count += dfs(pos+1, 0, 0) // horizontal
                                count += dfs(pos+1, 1, 0) // tromino top
                                count += dfs(pos+1, 0, 1) // tromino bottom
                            }
                            topFree==1 -> {
                                count += dfs(pos+1, 0, 0) // tromino
                                count += dfs(pos+1, 1, 0) // horizontal
                            }
                            bottomFree==1 -> {
                                count += dfs(pos+1, 0, 0) // tromino
                                count += dfs(pos+1, 0, 1) // horizontal
                            }
                        else -> {
                                count += dfs(pos+1, 1, 1) // skip
                            }
                        }

                        count = count % 1_000_000_007L
                    }
                    cache[pos][topFree][bottomFree] = count
                    count
                }
            }
        }
        return dfs(0, 1, 1).toInt()
    }

```

We can walk the board horizontally and monitor free cells. On each step, we can choose what figure to place. When end reached and there are no free cells, consider that a successful combination. Result depends only on the current position and on the top-bottom cell combination.* just do dfs+memo
* use array for a faster cache

Space: O(N), Time: O(N) - we only visit each column 3 times

# 23.12.2022
[309. Best Time to Buy and Sell Stock with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/) medium

[https://t.me/leetcode_daily_unstoppable/61](https://t.me/leetcode_daily_unstoppable/61)

[blog post](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/solutions/1522780/java-0ms-from-recursion-with-memo-to-iterative-o-n-time-and-o-1-memory/)

```kotlin 
    data class K(val a:Int, val b: Boolean, val c:Boolean)
    fun maxProfit(prices: IntArray): Int {
        val cache = mutableMapOf<K, Int>()
        fun dfs(pos: Int, canSell: Boolean, canBuy: Boolean): Int {
            return if (pos == prices.size) 0
                else cache.getOrPut(K(pos, canSell, canBuy), {
                    val profitSkip = dfs(pos+1, canSell, !canSell)
                    val profitSell = if (canSell) {prices[pos] + dfs(pos+1, false, false)} else 0
                    val profitBuy = if (canBuy) {-prices[pos] + dfs(pos+1, true, false)} else 0
                    maxOf(profitSkip, profitBuy, profitSell)
                })
        }
        return dfs(0, false, true)
    }

```

Progress from dfs solution to memo. DFS solution - just choose what to do in this step, go next, then compare results and peek max.

Space: O(N), Time: O(N)

# 22.12.2022
[834. Sum of Distances in Tree](https://leetcode.com/problems/sum-of-distances-in-tree/description/) hard

[https://t.me/leetcode_daily_unstoppable/60](https://t.me/leetcode_daily_unstoppable/60)

[blog post](https://leetcode.com/problems/sum-of-distances-in-tree/solutions/1443979/kotlin-java-2-dfs-diagramm-to-invent-the-change-root-equation/)

```kotlin 
    fun sumOfDistancesInTree(n: Int, edges: Array<IntArray>): IntArray {
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) -> 
            graph.getOrPut(from, { mutableListOf() }) += to
            graph.getOrPut(to, { mutableListOf() }) += from
        }
        val counts = IntArray(n) { 1 }
        val sums = IntArray(n) { 0 }
        fun distSum(pos: Int, visited: Int) {
            graph[pos]?.forEach {
                if (it != visited) {
                    distSum(it, pos)
                    counts[pos] += counts[it]
                    sums[pos] += counts[it] + sums[it]
                }
            }
        }
        fun dfs(pos: Int, visited: Int) {
            graph[pos]?.forEach {
                if (it != visited) {
                    sums[it] = sums[pos] - counts[it] + (n - counts[it])
                    dfs(it, pos)
                }
            }
        }
        distSum(0, -1)
        dfs(0, -1)
        return sums
    }

```

We can do the job for item #0, then we need to invent a formula to reuse some data when we change the node.

How to mathematically prove formula for a new sum:
![image](https://assets.leetcode.com/users/images/f7d1ffbc-7761-4cff-a219-58e1a433bd1c_1630765686.6135957.png)

![image.png](https://assets.leetcode.com/users/images/b2c81eba-e532-43cc-ae6a-6aec3eed57f9_1671730095.0767915.png)
Store count of children in a `counts` array, and sum of the distances to children in a `dist` array. In a first DFS traverse from a node 0 and fill the arrays. In a second DFS only modify `dist` based on previous computed `dist` value, using formula: `sum[curr] = sum[prev] - count[curr] + (N - count[curr])`

Space: O(N), Time: O(N)

# 21.12.2022
[886. Possible Bipartition](https://leetcode.com/problems/possible-bipartition/description/) medium

[https://t.me/leetcode_daily_unstoppable/59](https://t.me/leetcode_daily_unstoppable/59)

[blog post](https://leetcode.com/problems/possible-bipartition/solutions/2936306/kotlin-union-find/)

```kotlin 
fun possibleBipartition(n: Int, dislikes: Array<IntArray>): Boolean {
	val love = IntArray(n+1) { it }
	fun leader(x: Int): Int {
		var i = x
		while (love[i] != i) i = love[i]
		love[x] = i
		return i
	}
	val hate = IntArray(n+1) { -1 }
	dislikes.forEach { (one, two) ->
		val leaderOne = leader(one)
		val leaderTwo = leader(two)
		val enemyOfOne = hate[leaderOne]
		val enemyOfTwo = hate[leaderTwo]
		if (enemyOfOne != -1 && enemyOfOne == enemyOfTwo) return false
		if (enemyOfOne != -1) {
			love[leader(enemyOfOne)] = leaderTwo
		}
		if (enemyOfTwo != -1) {
			love[leader(enemyOfTwo)] = leaderOne
		}
		hate[leaderOne] = leaderTwo
		hate[leaderTwo] = leaderOne
	}
	return true
}

```

We need somehow to union people that hate the same people. We can do it making someone a leader of a group and make just leaders to hate each other.

Keep track of the leaders hating each other in the `hate` array, and people loving their leader in `love` array. (`love` array is basically a Union-Find).
* also use path compression for `leader` method

Space: O(N), Time: O(N) - adding to Union-Find is O(1) amortised

# 20.12.2022
[841. Keys and Rooms](https://leetcode.com/problems/keys-and-rooms/description/) medium

[https://t.me/leetcode_daily_unstoppable/58](https://t.me/leetcode_daily_unstoppable/58)

[blog post](https://leetcode.com/problems/keys-and-rooms/solutions/2932740/kotlin-dfs/)

```kotlin 
    fun canVisitAllRooms(rooms: List<List<Int>>): Boolean {
       val visited = hashSetOf(0)
       with(ArrayDeque<Int>()) {
           add(0)
           while(isNotEmpty()) {
               rooms[poll()].forEach {
                   if (visited.add(it)) add(it)
               }
           }
       }
       return visited.size == rooms.size
    }

```

We need to visit each room, and we have positions of the other rooms and a start position. This is a DFS problem.
Keep all visited rooms numbers in a hash set and check the final size. Other solution is to use boolean array and a counter of the visited rooms.

Space: O(N) - for queue and visited set, Time: O(N) - visit all the rooms once

# 19.12.2022
[1971. Find if Path Exists in Graph](https://leetcode.com/problems/find-if-path-exists-in-graph/description/) easy

[https://t.me/leetcode_daily_unstoppable/57](https://t.me/leetcode_daily_unstoppable/57)

[blog post](https://leetcode.com/problems/find-if-path-exists-in-graph/solutions/2928882/kotlin-bfs/)

```kotlin 
    fun validPath(n: Int, edges: Array<IntArray>, source: Int, destination: Int): Boolean {
        if (source == destination) return true
        val graph = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { (from, to) -> 
            graph.getOrPut(from, { mutableListOf() }).add(to)
            graph.getOrPut(to, { mutableListOf() }).add(from)
        }
        val visited = mutableSetOf<Int>()
        with(ArrayDeque<Int>()) {
            add(source)
            var depth = 0
            while(isNotEmpty() && ++depth < n) {
                repeat(size) {
                    graph[poll()]?.forEach {
                        if (it == destination) return true
                        if (visited.add(it)) add(it) 
                    }
                }
            }
        }
        return false
    }

```

BFS will do the job.
Make node to nodes map, keep visited set and use queue for BFS.
* also path can't be longer than n elements

Space: O(N), Time: O(N)

# 18.12.2022
[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/description/) medium

[https://t.me/leetcode_daily_unstoppable/55](https://t.me/leetcode_daily_unstoppable/55)

[blog post](https://leetcode.com/problems/daily-temperatures/solutions/2924323/kotlin-increasing-stack/)

```kotlin 
    fun dailyTemperatures(temperatures: IntArray): IntArray {
       val stack = Stack<Int>() 
       val res = IntArray(temperatures.size) { 0 }
       for (i in temperatures.lastIndex downTo 0) {
           while(stack.isNotEmpty() && temperatures[stack.peek()] <= temperatures[i]) stack.pop()
           if (stack.isNotEmpty()) {
               res[i] = stack.peek() - i
           }
           stack.push(i)
       }
       return res
    }

```

Intuitively, we want to go from the end of the array to the start and keep the maximum value. But, that doesn't work, because we must also store smaller numbers, as they are closer in distance.
For example, `4 3 5 6`, when we observe `4` we must compare it to `5`, not to `6`. So, we store not just max, but increasing max: `3 5 6`, and throw away all numbers smaller than current, `3 < 4` - pop().

We will iterate in reverse order, storing indexes in increasing by temperatures stack.

Space: O(N), Time: O(N)

# 17.12.2022
[150. Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/description/) medium

[https://t.me/leetcode_daily_unstoppable/54](https://t.me/leetcode_daily_unstoppable/54)

[blog post](https://leetcode.com/problems/evaluate-reverse-polish-notation/solutions/2922482/kotlin-stack/)

```kotlin 
    fun evalRPN(tokens: Array<String>): Int = with(Stack<Int>()) {
        tokens.forEach {
            when(it) {
                "+" -> push(pop() + pop())
                "-" -> push(-pop() + pop())
                "*" -> push(pop() * pop())
                "/" -> with(pop()) { push(pop()/this) }
                else -> push(it.toInt())
            }
        }
        pop()
    }

```

Reverse polish notations made explicitly for calculation using stack. Just execute every operation immediately using last two numbers in the stack and push the result.
* be aware of the order of the operands

Space: O(N), Time: O(N)

# 16.12.2022
[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/description/) easy

[https://t.me/leetcode_daily_unstoppable/53](https://t.me/leetcode_daily_unstoppable/53)

[blog post](https://leetcode.com/problems/implement-queue-using-stacks/solutions/2918693/kotlin-head-tail/)

```kotlin 
class MyQueue() {
	val head = Stack<Int>()
	val tail = Stack<Int>()

	//  []       []
	//  1 2 3 4 -> 4 3 2 - 1
	//  5         4 3 2
	//            4 3 2 5
	fun push(x: Int) {
		head.push(x)
	}

	fun pop(): Int {
		peek()

		return tail.pop()
	}

	fun peek(): Int {
		if (tail.isEmpty()) while(head.isNotEmpty()) tail.push(head.pop())

		return tail.peek()
	}

	fun empty(): Boolean = head.isEmpty() && tail.isEmpty()

}

```

One stack for the head of the queue and other for the tail.
When we need to do `pop` we first drain from one stack to another, so items order will be restored.
* we can skip rotation on push if we fill tail only when its empty

Space: O(1), Time: O(1)

# 15.12.2022
[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/description/) medium

[https://t.me/leetcode_daily_unstoppable/52](https://t.me/leetcode_daily_unstoppable/52)

[blog post](https://leetcode.com/problems/longest-common-subsequence/solutions/2915134/kotlin-dfs-memo/)

```kotlin 
    fun longestCommonSubsequence(text1: String, text2: String): Int {
        val cache = Array(text1.length + 1) { IntArray(text2.length + 1) { -1 } }
        fun dfs(pos1: Int, pos2: Int): Int {
            if (pos1 == text1.length) return 0
            if (pos2 == text2.length) return 0
            val c1 = text1[pos1]
            val c2 = text2[pos2]
            if (cache[pos1][pos2] != -1) return cache[pos1][pos2]
            val res = if (c1 == c2) {
                    1 + dfs(pos1 + 1, pos2 + 1)
                } else {
                    maxOf(dfs(pos1, pos2+1), dfs(pos1+1, pos2))
                }
            cache[pos1][pos2] = res
            return res
        }
        return dfs(0, 0)
    }

```

We can walk the two strings simultaneously and compare their chars. If they are the same, the optimal way will be to use those chars and continue exploring next. If they are not, we have two choices: use the first char and skip the second or skip the first but use the second.
Also, observing our algorithm we see, the result so far is only dependent of the positions from which we begin to search (and all the remaining characters). And also see that the calls are repetitive. That mean we can cache the result. (meaning this is a dynamic programming solution).
Use depth first search by starting positions and memoize results in a two dimension array. Another approach will be bottom up iteration and filling the same array.

Space: O(N^2), Time: O(N^2)

# 14.12.2022
[198. House Robber](https://leetcode.com/problems/house-robber/description/) medium

[https://t.me/leetcode_daily_unstoppable/51](https://t.me/leetcode_daily_unstoppable/51)

[blog post](https://leetcode.com/problems/house-robber/solutions/2911816/kotlin-dfs-memo/)

```kotlin 
    fun rob(nums: IntArray): Int {
        val cache = mutableMapOf<Int, Int>()
        fun dfs(pos: Int): Int {
            if (pos > nums.lastIndex) return 0
            return cache.getOrPut(pos) {
                maxOf(nums[pos] + dfs(pos+2), dfs(pos+1))
            }
        } 
        return dfs(0)
    }

```

Exploring each house one by one we can make a decision to rob or not to rob.
The result is only depends on our current position (and all houses that are remaining to rob) and decision, so we can memoize it based on position.

We can use memoization or walk houses bottom up.

Space: O(N), Time: O(N)

# 13.12.2022
[931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/description/) medium

[https://t.me/leetcode_daily_unstoppable/50](https://t.me/leetcode_daily_unstoppable/50)

[blog post](https://leetcode.com/problems/minimum-falling-path-sum/solutions/2908108/kotlin-running-sum/)

```kotlin 
    fun minFallingPathSum(matrix: Array<IntArray>): Int {
       for (y in matrix.lastIndex-1 downTo 0) {
           val currRow = matrix[y]
           val nextRow = matrix[y+1]
           for (x in 0..matrix[0].lastIndex) {
               val left = if (x > 0) nextRow[x-1] else Int.MAX_VALUE
               val bottom = nextRow[x]
               val right = if (x < matrix[0].lastIndex) nextRow[x+1] else Int.MAX_VALUE
               val minSum = currRow[x] + minOf(left, bottom, right)
               currRow[x] = minSum
           }
       } 
       return matrix[0].min()!!
    }

```

There is only three ways from any cell to it's siblings. We can compute all three paths sums for all cells in a row so far. And then choose the smallest.
Iterate over rows and compute prefix sums of current + minOf(left min sum, bottom min sum, right min sum)

Space: O(N), Time: O(N^2)

# 12.12.2022
[70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/description/) easy

[https://t.me/leetcode_daily_unstoppable/49](https://t.me/leetcode_daily_unstoppable/49)

[blog post](https://leetcode.com/problems/climbing-stairs/solutions/2904774/kotlin-dfs-memo/)

```kotlin 
    val cache = mutableMapOf<Int, Int>()
    fun climbStairs(n: Int): Int = when {
        n < 1  -> 0
        n == 1 -> 1
        n == 2 -> 2
        else -> cache.getOrPut(n) {
            climbStairs(n-1) + climbStairs(n-2)
        }
    }

```

You can observe that result is only depend on input n. And also that result(n) = result(n-1) + result(n-2).
Just use memoization for storing already solved inputs.

Space: O(N), Time: O(N)

# 11.12.2022
[124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/) hard

[https://t.me/leetcode_daily_unstoppable/48](https://t.me/leetcode_daily_unstoppable/48)

[blog post](https://leetcode.com/problems/binary-tree-maximum-path-sum/solutions/2900498/kotlin-very-bad-problem-definition/)

```kotlin 
    fun maxPathSum(root: TreeNode?): Int {
        fun dfs(root: TreeNode): Pair<Int, Int> {
            val lt = root.left
            val rt = root.right
            if (lt == null && rt == null) return root.`val` to root.`val`
            if (lt == null || rt == null) {
                val sub = dfs(if (lt == null) rt else lt)
                val currRes = root.`val` + sub.second
                val maxRes = maxOf(sub.first, currRes, root.`val`)
                val maxPath = maxOf(root.`val`, root.`val` + sub.second)
                return maxRes to maxPath
            } else {
                val left = dfs(root.left)
                val right = dfs(root.right)
                val currRes1 = root.`val` + left.second + right.second
                val currRes2 = root.`val`
                val currRes3 = root.`val` + left.second
                val currRes4 = root.`val` + right.second
                val max1 = maxOf(currRes1, currRes2)
                val max2 = maxOf(currRes3, currRes4)
                val maxRes = maxOf(left.first, right.first, maxOf(max1, max2))
                val maxPath = maxOf(root.`val`, root.`val` + maxOf(left.second, right.second))
                return maxRes to maxPath
            }
        }
        return if (root == null) 0 else dfs(root).first
    }

```

Space: O(logN), Time: O(N)

# 10.12.2022
[1339. Maximum Product of Splitted Binary Tree](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/description/) medium

[https://t.me/leetcode_daily_unstoppable/47](https://t.me/leetcode_daily_unstoppable/47)

[blog post](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/solutions/2896607/kotlin-two-dfs/)

```kotlin

    fun maxProduct(root: TreeNode?): Int {
        fun sumDfs(root: TreeNode?): Long {
            return if (root == null) 0L
            else with(root) { `val`.toLong() + sumDfs(left) + sumDfs(right) }
        }
        val total = sumDfs(root)
        fun dfs(root: TreeNode?) : Pair<Long, Long> {
            if (root == null) return Pair(0,0)
            val left = dfs(root.left)
            val right = dfs(root.right)
            val sum = left.first + root.`val`.toLong() + right.first
            val productLeft = left.first * (total - left.first) 
            val productRight = right.first * (total - right.first)
            val prevProductMax = maxOf(right.second, left.second)
            return sum to maxOf(productLeft, productRight, prevProductMax)
        }
        return (dfs(root).second % 1_000_000_007L).toInt()
    }

```

Just iterate over all items and compute all products.
We need to compute total sum before making the main traversal.

Space: O(logN), Time: O(N)

# 9.12.2022
[1026. Maximum Difference Between Node and Ancestor](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/description/) medium

[https://t.me/leetcode_daily_unstoppable/46](https://t.me/leetcode_daily_unstoppable/46)

[blog post](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/solutions/2894948/kotlin-dfs/)

```kotlin

    fun maxAncestorDiff(root: TreeNode?): Int {
        root?: return 0

        fun dfs(root: TreeNode, min: Int = root.`val`, max: Int = root.`val`): Int {
            val v = root.`val`
            val currDiff = maxOf(Math.abs(v - min), Math.abs(v - max))
            val currMin = minOf(min, v)
            val currMax = maxOf(max, v)
            val leftDiff = root.left?.let { dfs(it, currMin, currMax) } ?: 0
            val rightDiff = root.right?.let { dfs(it, currMin, currMax) } ?: 0
            return maxOf(currDiff, leftDiff, rightDiff)
        }
        
        return dfs(root)
    }

```

Based on math we can assume, that max difference is one of the two: (curr - max so far) or (curr - min so far).
Like, for example, let our curr value be `3`, and from all visited we have min `0` and max `7`.

```

 0--3---7

```

* we can write helper recoursive method and compute max and min so far

Space: O(logN), Time: O(N)

# 8.12.2022
[872. Leaf-Similar Trees](https://leetcode.com/problems/leaf-similar-trees/solutions/) easy

[https://t.me/leetcode_daily_unstoppable/45](https://t.me/leetcode_daily_unstoppable/45)

```kotlin

    fun leafSimilar(root1: TreeNode?, root2: TreeNode?): Boolean {
        fun dfs(root: TreeNode?): List<Int> {
            return when {
                root == null -> listOf()
                root.left == null && root.right == null -> listOf(root.`val`)
                else -> dfs(root.left) + dfs(root.right)
            }
        }
        
        return dfs(root1) == dfs(root2)
    }

```

There is only 200 items, so we can concatenate lists.
One optimization would be to collect only first tree and just compare it to the second tree while doing the inorder traverse.

Space: O(N), Time: O(N)

# 7.12.2022
[938. Range Sum of BST](https://leetcode.com/problems/range-sum-of-bst/description/) easy

[https://t.me/leetcode_daily_unstoppable/44](https://t.me/leetcode_daily_unstoppable/44)

```kotlin

    fun rangeSumBST(root: TreeNode?, low: Int, high: Int): Int =
	if (root == null) 0 else
		with(root) {
			(if (`val` in low..high) `val` else 0) +
				(if (`val` < low) 0 else rangeSumBST(left, low, high)) +
				(if (`val` > high) 0 else rangeSumBST(right, low, high))
		}

```

* be careful with ternary operations, better wrap them in a brackets

Space: O(log N), Time: O(R), r - is a range [low, high]

# 6.12.2022
[328. Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/description/) medium

[https://t.me/leetcode_daily_unstoppable/43](https://t.me/leetcode_daily_unstoppable/43)

```kotlin

       // 1 2
    fun oddEvenList(head: ListNode?): ListNode? {
       var odd = head //1
       val evenHead = head?.next
       var even = head?.next //2
       while(even!=null) { //2
           val oddNext = odd?.next?.next //null
           val evenNext = even?.next?.next //null
           odd?.next = oddNext // 1->null
           even?.next = evenNext //2->null
           if (oddNext != null) odd = oddNext //
           even = evenNext // null
       }
       odd?.next = evenHead // 1->2
       return head //1->2->null
    }

```

* be careful and store evenHead in a separate variable

Space: O(1), Time: O(n)

# 5.12.2022
[876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) easy

[https://t.me/leetcode_daily_unstoppable/42](https://t.me/leetcode_daily_unstoppable/42)

```kotlin

  fun middleNode(head: ListNode?, fast: ListNode? = head): ListNode? =
        if (fast?.next == null) head else middleNode(head?.next, fast?.next?.next)

```

* one-liner, but in the interview (or production) I would prefer to write a loop

Space: O(n), Time: O(n)

# 4.12.2022
[2256. Minimum Average Difference](https://leetcode.com/problems/minimum-average-difference/) medium

[https://t.me/leetcode_daily_unstoppable/41](https://t.me/leetcode_daily_unstoppable/41)

```kotlin

    fun minimumAverageDifference(nums: IntArray): Int {
        var sum = 0L
        nums.forEach { sum += it.toLong() }
        var leftSum = 0L
        var min = Long.MAX_VALUE
        var minInd = 0
        for (i in 0..nums.lastIndex) {
            val leftCount = (i+1).toLong()
            leftSum += nums[i].toLong()
            val front = leftSum/leftCount
            val rightCount = nums.size.toLong() - leftCount
            val rightSum = sum - leftSum
            val back = if (rightCount == 0L) 0L else rightSum/rightCount
            val diff = Math.abs(front - back)
            if (diff < min) {
                min = diff
                minInd = i
            }
        }
        return minInd
    }

```

### Intuition

Two pointers, one for even, one for odd indexes.
### Approach

To avoid mistakes you need to be verbose, and don't skip operations:
* store evenHead in a separate variable
* don't switch links before both pointers jumped
* don't make odd pointer null
* try to run for simple input `1->2->null` by yourself

  
Space: O(1), Time: O(n)

# 3.12.2022
[451. Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/) medium

[https://t.me/leetcode_daily_unstoppable/40](https://t.me/leetcode_daily_unstoppable/40)

```kotlin

    fun frequencySort(s: String): String =
        s.groupBy { it }
        .values
        .map { it to it.size }
        .sortedBy { -it.second }
        .map { it.first }
        .flatten()
        .joinToString("")

```

Very simple task, can be written in a functional style.
Space: O(n), Time: O(n)

# 2.12.2022
[https://leetcode.com/problems/determine-if-two-strings-are-close/](https://leetcode.com/problems/determine-if-two-strings-are-close/) medium

[https://t.me/leetcode_daily_unstoppable/39](https://t.me/leetcode_daily_unstoppable/39)

```kotlin

    // cabbba -> c aa bbb -> 1 2 3 
    // a bb ccc -> 1 2 3
    // uau
    // ssx
    fun closeStrings(word1: String, word2: String, 
         f: (String) -> List<Int> = { it.groupBy { it }.values.map { it.size }.sorted() }
    ): Boolean = f(word1) == f(word2) && word1.toSet() == word2.toSet()

```

That is a simple task, you just need to know what exactly you asked for.
Space: O(n), Time: O(n)

# 1.12.2022
[1704. Determine if String Halves Are Alike](https://leetcode.com/problems/determine-if-string-halves-are-alike/) easy

[https://t.me/leetcode_daily_unstoppable/38](https://t.me/leetcode_daily_unstoppable/38)

```kotlin

    fun halvesAreAlike(s: String): Boolean {
        val vowels = setOf('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        var c1 = 0
        var c2 = 0
        s.forEachIndexed { i, c -> 
            if (c in vowels) {
                if (i < s.length / 2) c1++ else c2++
            }
        }
        return c1 == c2
    }

```

Just do what is asked.

O(N) time, O(1) space

# 30.11.2022
[1207. Unique Number of Occurrences](https://leetcode.com/problems/unique-number-of-occurrences/) easy

[https://t.me/leetcode_daily_unstoppable/36](https://t.me/leetcode_daily_unstoppable/36)

```kotlin

fun uniqueOccurrences(arr: IntArray): Boolean {
	val counter = mutableMapOf<Int, Int>()
	arr.forEach { n -> counter[n] = 1 + (counter[n] ?: 0) }
	val freq = mutableSetOf<Int>()
	return !counter.values.any { count -> !freq.add(count) }
}

```

Nothing interesting, just count and filter.

O(N) time, O(N) space

# 29.11.2022
[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) medium

[https://t.me/leetcode_daily_unstoppable/35](https://t.me/leetcode_daily_unstoppable/35)

```kotlin

class RandomizedSet() {
    val rnd = Random(0)
    val list = mutableListOf<Int>()
    val vToInd = mutableMapOf<Int, Int?>()
    fun insert(v: Int): Boolean {
        if (!vToInd.contains(v)) {
            vToInd[v] = list.size
            list.add(v)
            return true
        }
        return false
    }
    fun remove(v: Int): Boolean {
        val ind = vToInd[v] ?: return false
        val prevLast = list[list.lastIndex]
        list[ind] = prevLast
        vToInd[prevLast] = ind
        list.removeAt(list.lastIndex)
        vToInd.remove(v)
        return true
    }
    fun getRandom(): Int = list[rnd.nextInt(list.size)]
}

```

The task is simple, one trick is to remove elements from the end of the list, and replacing item with the last one.
Some thoughts:
* don't optimize lines of code, that can backfire. You can use syntax sugar, clever operations inlining, but also can shoot in the foot.

O(1) time, O(N) space

# 28.11.2022
[2225. Find Players With Zero or One Losses](https://leetcode.com/problems/find-players-with-zero-or-one-losses/) medium

[https://t.me/leetcode_daily_unstoppable/34](https://t.me/leetcode_daily_unstoppable/34)

```kotlin

    fun findWinners(matches: Array<IntArray>): List<List<Int>> {
        val winners = mutableMapOf<Int, Int>()
        val losers = mutableMapOf<Int, Int>()
        matches.forEach { (w, l) ->
            winners[w] = 1 + (winners[w]?:0)
            losers[l] = 1 + (losers[l]?:0)
        }
        return listOf(
            winners.keys
                .filter { !losers.contains(it) }
                .sorted(),
            losers
                .filter { (k, v) -> v == 1 }
                .map { (k, v) -> k}
                .sorted()
        )
    }

```

Just do what is asked.

O(NlogN) time, O(N) space

# 27.11.2022
[446. Arithmetic Slices II - Subsequence](https://leetcode.com/problems/arithmetic-slices-ii-subsequence/) hard

[https://t.me/leetcode_daily_unstoppable/33](https://t.me/leetcode_daily_unstoppable/33)

```kotlin

    fun numberOfArithmeticSlices(nums: IntArray): Int {
        // 0 1 2 3 4 5 
        // 1 2 3 1 2 3                diff = 1
        //   ^     ^ *                dp[5][diff] = 
        //   |     |  \__ curr        1 + dp[1][diff] +
        //  prev   |                  1 + dp[4][diff]
        //        prev
        // 
        val dp = Array(nums.size) { mutableMapOf<Long, Long> () }
        for (curr in 0..nums.lastIndex) {
            for (prev in 0 until curr) {
                val diff = nums[curr].toLong() - nums[prev].toLong()
                dp[curr][diff] = 1 + (dp[curr][diff]?:0L) + (dp[prev][diff]?:0L)
            }
        }
        return dp.map { it.values.sum()!! }.sum().toInt() - (nums.size)*(nums.size-1)/2
    }

```

dp[i][d] is the number of subsequences in range [0..i] with difference = d

```kotlin

array: "1 2 3 1 2 3"
For items  1  2  curr = 2:
diff = 1,  dp = 1
For items  1  2  3  curr = 3:
diff = 2,  dp = 1
diff = 1,  dp = 2
For items  1  2  3  1  curr = 1:
diff = 0,  dp = 1
diff = -1,  dp = 1
diff = -2,  dp = 1
For items  1  2  3  1  2  curr = 2:
diff = 1,  dp = 2
diff = 0,  dp = 1
diff = -1,  dp = 1
For items  1  2  3  1  2  3  curr = 3:
diff = 2,  dp = 2
diff = 1,  dp = 5
diff = 0,  dp = 1

```

and finally, we need to subtract all the sequences of length 2 and 1,
count of them is (n)*(n-1)/2

O(N^2) time, O(N^2) space

# 26.11.2022
[1235. Maximum Profit in Job Scheduling](https://leetcode.com/problems/maximum-profit-in-job-scheduling/) hard

[https://t.me/leetcode_daily_unstoppable/32](https://t.me/leetcode_daily_unstoppable/32)

```kotlin

    fun jobScheduling(startTime: IntArray, endTime: IntArray, profit: IntArray): Int {
        val n = startTime.size
        val inds = Array<Int>(n) { it }
        inds.sortWith (Comparator<Int> { a, b -> 
            if (startTime[a] == startTime[b])
                endTime[a] - endTime[b]
            else
                startTime[a] - startTime[b]
        })
        val maxProfit = IntArray(n) { 0 }
        maxProfit[n-1] = profit[inds[n-1]]
        for (i in n-2 downTo 0) {
            val ind = inds[i]
            val end = endTime[ind]
            val prof = profit[ind]
            
            var lo = l + 1
            var hi = n - 1
            var nonOverlapProfit = 0
            while (lo <= hi) {
                val mid = lo + (hi - lo) / 2
                if (end <= startTime[inds[mid]]) {
                    nonOverlapProfit = maxOf(nonOverlapProfit, maxProfit[mid])
                    hi = mid - 1
                } else lo = mid + 1
            }
            maxProfit[i] = maxOf(prof + nonOverlapProfit, maxProfit[i+1])
        }
        return maxProfit[0]
    }

```

Use the hints from the description.
THis cannot be solved greedily, because you need to find next non-overlapping job.
Dynamic programming equation: from last job to the current, result is max of next result and current + next non-overlapping result.

```

f(i) = max(f(i+1), profit[i] + f(j)), where j is the first non-overlapping job after i.

```

Also, instead of linear search for non overlapping job, use binary search.

O(NlogN) time, O(N) space

# 25.11.2022
[907. Sum of Subarray Minimums](https://leetcode.com/problems/sum-of-subarray-minimums/) medium

```kotlin

    data class V(val v: Int, val count: Int)
    fun sumSubarrayMins(arr: IntArray): Int {
        val M = 1_000_000_007
        // 1 2 3 4 2 2 3 4
        //  1 2 3 2 2 2 3
        //   1 2 2 2 2 2
        //    1 2 2 2 2
        //     1 2 2 2
        //      1 2 2
        //       1 2
        //        1
        // f(1) = 1
        // f(2) = 2>1 ? f(1) + [1, 2]
        // f(3) = 3>2 ? f(2) + [1, 2, 3]
        // f(4) = 4>3 ? f(3) + [1, 2, 3, 4]
        // f(2) = 2<4 ? f(4) + [1, 2, 2, 2, 2] (1, 2, 3, 4 -> 3-2, 4-2, +2)
        // f(2) = 2=2 ? f(2) + [1, 2, 2, 2, 2, 2]
        // f(3) = 3>2 ? f(2) + [1, 2, 2, 2, 2, 2, 3]
        // f(4) = 4>3 ? f(3) + [1, 2, 2, 2, 2, 2, 3, 4]
        // 3 1 2 4    f(3) = 3    sum = 3  stack: [3]
        //  1 1 2     f(1): 3 > 1 , remove V(3,1), sum = sum - 3 + 1*2= 2, f=3+2=5, [(1,2)]
        //   1 1      f(2): 2>1, sum += 2 = 4, f+=4=9
        //    1       f(4): 4>2, sum+=4=8, f+=8=17
        val stack = Stack<V>()
        var f = 0
        var sum = 0
        arr.forEach { n ->
            var countRemoved = 0
            while (stack.isNotEmpty() && stack.peek().v > n) {
                val v = stack.pop()
                countRemoved += v.count
                var removedSum = (v.v*v.count) % M
                if (removedSum < 0) removedSum = M + removedSum
                sum = (sum - removedSum) % M
                if (sum < 0) sum = sum + M
            }
            val count = countRemoved + 1
            stack.add(V(n, count))
            sum = (sum + (n * count) % M) % M
            f = (f + sum) % M
            
        }
        return f
    }

```

First attempt is to build an N^2 tree of minimums, comparing adjacent elements row by row and finding a minimum.
That will take O(N^2) time and gives TLE. 
Next observe that there is a repetition of the results if we computing result for each new element:
result = previous result + some new elements.
That new elements are also have a law of repetition: 
sum = current element + if (current element < previous element) count of previous elements * current element else previous sum
We can use a stack to keep lowest previous elements, all values in stack must be less than current element.

O(N) time, O(N) space

# 24.11.2022
[79. Word Search](https://leetcode.com/problems/word-search/) medium

```kotlin

    fun exist(board: Array<CharArray>, word: String): Boolean {
        fun dfs(y: Int, x: Int, pos: Int): Boolean {
            if (pos == word.length) return true
            if (y < 0 || x < 0 || y == board.size || x == board[0].size) return false
            val c = board[y][x]
            if (c != word[pos]) return false
            board[y][x] = '.'
            val res = dfs(y-1, x, pos+1)
                   || dfs(y+1, x, pos+1)
                   || dfs(y, x-1, pos+1)
                   || dfs(y, x+1, pos+1)
            board[y][x] = c
            return res
        }
        for (y in 0..board.lastIndex) {
            for (x in 0..board[0].lastIndex) {
                if (dfs(y, x, 0)) return true
            }
        }
        return false
    }

```

We can brute force this problem. Backtracking help to preserve memory.

Complexity: O(M*N*W)
Memory: O(W)

# 23.11.2022
[https://leetcode.com/problems/valid-sudoku/](https://leetcode.com/problems/valid-sudoku/) medium

```kotlin

    fun isValidSudoku(board: Array<CharArray>): Boolean {
        val cell9 = arrayOf(0 to 0, 0 to 1, 0 to 2, 
                            1 to 0, 1 to 1, 1 to 2, 
                            2 to 0, 2 to 1, 2 to 2)
        val starts = arrayOf(0 to 0, 0 to 3, 0 to 6, 
                             3 to 0, 3 to 3, 3 to 6, 
                             6 to 0, 6 to 3, 6 to 6)
        return !starts.any { (sy, sx) ->
                val visited = HashSet<Char>()
                cell9.any { (dy, dx) ->
                    val c = board[sy+dy][sx+dx]
                    c != '.' && !visited.add(c)
                }
            } && !board.any { row -> 
                val visited = HashSet<Char>()
                row.any { it != '.' && !visited.add(it) }
            } && !(0..8).any { x ->
                val visited = HashSet<Char>()
                (0..8).any { board[it][x] != '.' && !visited.add(board[it][x]) }
            }
    }

```

This is an easy problem, just do what is asked.

Complexity: O(N)
Memory: O(N), N = 81, so it O(1)

# 22.11.2022
[https://leetcode.com/problems/perfect-squares/](https://leetcode.com/problems/perfect-squares/) medium

```kotlin

    val cache = mutableMapOf<Int, Int>()
    fun numSquares(n: Int): Int {
        if (n < 0) return -1
        if (n == 0) return 0
        if (cache[n] != null) return cache[n]!!
        var min = Int.MAX_VALUE
        for (x in Math.sqrt(n.toDouble()).toInt() downTo 1) {
            val res = numSquares(n - x*x)
            if (res != -1) {
                min = minOf(min, 1 + res)
            }
        }
        if (min == Int.MAX_VALUE) min = -1
        cache[n] = min
        return min
    }

```

The problem gives stable answers for any argument n. 
So, we can use memoization technique and search from the biggest square to the smallest one.

Complexity: O(Nsqrt(N))
Memory: O(N)

# 21.11.2022
[https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/](https://leetcode.com/problems/nearest-exit-from-entrance-in-maze/) medium

```

    fun nearestExit(maze: Array<CharArray>, entrance: IntArray): Int {
        val queue = ArrayDeque<Pair<Int, Int>>()
        queue.add(entrance[1] to entrance[0])
        maze[entrance[0]][entrance[1]] = 'x'
        var steps = 1
        val directions = intArrayOf(-1, 0, 1, 0, -1)
        while(queue.isNotEmpty()) {
            repeat(queue.size){
                val (x, y) = queue.poll()
                for (i in 1..directions.lastIndex) {
                    val nx = x + directions[i-1]
                    val ny = y + directions[i]
                    if (nx in 0..maze[0].lastIndex &&
                            ny in 0..maze.lastIndex &&
                            maze[ny][nx] == '.') {
                        if (nx == 0 || 
                                ny == 0 || 
                                nx == maze[0].lastIndex || 
                                ny == maze.lastIndex) return steps
                        maze[ny][nx] = 'x'
                        queue.add(nx to ny)
                    }
                }
            }
            steps++
        }
        
        return -1
    }

```

Just do BFS.
* we can modify input matrix, so we can use it as visited array

Complexity: O(N), N - number of cells in maze
Memory: O(N)

# 20.11.2022
[https://leetcode.com/problems/basic-calculator/](https://leetcode.com/problems/basic-calculator/) hard

```

    fun calculate(s: String): Int {
        var i = 0
        var sign = 1
        var eval = 0
        while (i <= s.lastIndex) {
            val chr = s[i]
            if (chr == '(') {
                //find the end
                var countOpen = 0
                for (j in i..s.lastIndex) {
                    if (s[j] == '(') countOpen++
                    if (s[j] == ')') countOpen--
                    if (countOpen == 0) {
                        //evaluate substring
                        eval += sign * calculate(s.substring(i+1, j)) // [a b)
                        sign = 1
                        i = j
                        break
                    }
                }
            } else if (chr == '+') {
                sign = 1
            } else if (chr == '-') {
                sign = -1
            } else if (chr == ' ') {
                //nothing
            } else {
                var num = (s[i] - '0').toInt()
                for (j in (i+1)..s.lastIndex) {
                    if (s[j].isDigit()) {
                        num = num * 10 + (s[j] - '0').toInt()
                        i = j
                    } else  break
                }
                eval += sign * num
                sign = 1
            }
            i++
        }
        return eval
    }

```

This is a classic calculator problem, nothing special.
* be careful with the indexes

Complexity: O(N)
Memory: O(N), because of the recursion, worst case is all the input is brackets

# 19.11.2022
[https://leetcode.com/problems/erect-the-fence/](https://leetcode.com/problems/erect-the-fence/) hard

```

    fun outerTrees(trees: Array<IntArray>): Array<IntArray> {
        if (trees.size <= 3) return trees
        trees.sortWith(Comparator { a, b -> if (a[0]==b[0]) a[1]-b[1] else a[0] - b[0]} )
        fun cmp(a: IntArray, b: IntArray, c: IntArray): Int {
            val xab = b[0] - a[0]
            val yab = b[1] - a[1]
            val xbc = c[0] - b[0]
            val ybc = c[1] - b[1]
            return xab*ybc - yab*xbc
        }
        val up = mutableListOf<IntArray>()
        val lo = mutableListOf<IntArray>()
        trees.forEach { curr ->
            while(up.size >= 2 && cmp(up[up.size-2], up[up.size-1], curr) < 0) up.removeAt(up.lastIndex)
            while(lo.size >= 2 && cmp(lo[lo.size-2], lo[lo.size-1], curr) > 0) lo.removeAt(lo.lastIndex)
            up.add(curr)
            lo.add(curr)
        }
        return (up+lo).distinct().toTypedArray()
    }

```

This is an implementation of the [Andrew's monotonic chain](https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain) algorithm.
* need to remember vector algebra equation for ccw (counter clockwise) check (see [here](https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon))
* don't forget to sort by x and then by y

Complexity: O(Nlog(N))
Memory: O(N)

# 18.11.2022
[https://leetcode.com/problems/ugly-number/](https://leetcode.com/problems/ugly-number/) easy

```

    fun isUgly(n: Int): Boolean {
        if (n <= 0) return false
        var x = n
        while(x%2==0) x = x/2
        while(x%3==0) x = x/3
        while(x%5==0) x = x/5
        return x == 1
    }

```

There is also a clever math solution, but I don't understand it yet.

Complexity: O(log(n))
Memory: O(1)

# 17.11.2022
[https://leetcode.com/problems/rectangle-area/](https://leetcode.com/problems/rectangle-area/) middle

```kotlin

class Solution {
    class P(val x: Int, val y: Int)
    class Rect(val l: Int, val t: Int, val r: Int, val b: Int) {
        val corners = arrayOf(P(l, t), P(l, b), P(r, t), P(r, b))
        val s = (r - l) * (t - b)
        fun contains(p: P) = p.x in l..r && p.y in b..t
        fun intersect(o: Rect): Rect {
            val allX = intArrayOf(l, r, o.l, o.r).apply { sort() }
            val allY = intArrayOf(b, t, o.b, o.t).apply { sort() }
            val r = Rect(allX[1], allY[2], allX[2], allY[1])
            return if (r.corners.all { contains(it) && o.contains(it)}) 
                r else Rect(0,0,0,0)
        }
    }
    
    fun computeArea(ax1: Int, ay1: Int, ax2: Int, ay2: Int, bx1: Int, by1: Int, bx2: Int, by2: Int): Int {
        val r1 = Rect(ax1, ay2, ax2, ay1)
        val r2 = Rect(bx1, by2, bx2, by1)
        return r1.s + r2.s -  r1.intersect(r2).s
    }
}

```

This is an OOP problem. One trick to write intersection function is to notice that all corners of intersection rectangle
must be inside both rectangles. Also, intersection rectangle formed from middle coordinates of all corners sorted by x and y.

Complexity: O(1)
Memory: O(1)

# 16.11.2022
[https://leetcode.com/problems/guess-number-higher-or-lower/](https://leetcode.com/problems/guess-number-higher-or-lower/) easy

```kotlin

    override fun guessNumber(n:Int):Int {
       var lo = 1
       var hi = n
       while(lo <= hi) {
           val pick = lo + (hi - lo)/2
           val answer = guess(pick)
           if (answer == 0) return pick
           if (answer == -1) hi = pick - 1
           else lo = pick + 1
       }
       return lo
    }

```

This is a classic binary search algorithm. 
The best way of writing it is:
* use safe mid calculation (lo + (hi - lo)/2)
* use lo <= hi instead of lo < hi and mid+1/mid-1 instead of mid
  
Complexity: O(log(N))
Memory: O(1)

# 15.11.2022
[https://leetcode.com/problems/count-complete-tree-nodes/](https://leetcode.com/problems/count-complete-tree-nodes/) medium

```

       x
     *   x
   *   *   x
 *   x   *   x
* x x x x * x x
          \
          on each node we can check it's left and right depths
          this only takes us O(logN) time on each step
          there are logN steps in total (height of the tree)
          so the total time complexity is O(log^2(N))

```

```kotlin

    fun countNodes(root: TreeNode?): Int {
        var hl = 0
        var node = root  
        while (node != null) {
            node = node.left
            hl++
        }
        var hr = 0
        node = root  
        while (node != null) {
            node = node.right
            hr++
        }
        return when {
            hl == 0 -> 0 
            hl == hr -> (1 shl hl) - 1
            else -> 1  + 
            (root!!.left?.let {countNodes(it)}?:0) +
            (root!!.right?.let {countNodes(it)}?:0)
        }
    }

```

Complexity: O(log^2(N))
Memory: O(logN)

# 14.11.2022
[https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/](https://leetcode.com/problems/most-stones-removed-with-same-row-or-column/) medium

From observing the problem, we can see, that the task is in fact is to find an isolated islands:

```

        // * 3 *         * 3 *        * * *
        // 1 2 *    ->   * * *   or   1 * *
        // * * 4         * * 4        * * 4

        // * 3 *         * * *
        // 1 2 5    ->   * * *
        // * * 4         * * 4

```

```kotlin

    fun removeStones(stones: Array<IntArray>): Int {
        val uf = IntArray(stones.size) { it }
        var rootsCount = uf.size
        fun root(a: Int): Int {
            var x = a
            while (uf[x] != x) x = uf[x]
            return x
        }
        fun union(a: Int, b: Int) {
           val rootA = root(a) 
           val rootB = root(b)
           if (rootA != rootB) {
               uf[rootA] = rootB
               rootsCount--
           }
        }
        val byY = mutableMapOf<Int, MutableList<Int>>()
        val byX = mutableMapOf<Int, MutableList<Int>>()
        stones.forEachIndexed { i, st ->
            byY.getOrPut(st[0], { mutableListOf() }).add(i)
            byX.getOrPut(st[1], { mutableListOf() }).add(i)
        }
        byY.values.forEach { list ->
            if (list.size > 1) 
                for (i in 1..list.lastIndex) union(list[0], list[i])
        }
        byX.values.forEach { list ->
            if (list.size > 1) 
                for (i in 1..list.lastIndex) union(list[0], list[i])
        }
        return stones.size - rootsCount
    }

```

Complexity: O(N)
Memory: O(N)

# 13.11.2022
[https://leetcode.com/problems/reverse-words-in-a-string/](https://leetcode.com/problems/reverse-words-in-a-string/) medium

A simple trick: reverse all the string, then reverse each word.

```kotlin

    fun reverseWords(s: String): String {
        val res = StringBuilder()
        val curr = Stack<Char>()
        (s.lastIndex downTo 0).forEach { i ->
            val c = s[i]
            if (c in '0'..'z') curr.push(c)
            else if (curr.isNotEmpty()) {
                if (res.length > 0) res.append(' ')
                while (curr.isNotEmpty()) res.append(curr.pop())
            }
        }
        if (curr.isNotEmpty() && res.length > 0) res.append(' ')
        while (curr.isNotEmpty()) res.append(curr.pop())
        return res.toString()
    }

```

Complexity: O(N)
Memory: O(N) - there is no O(1) solution for string in JVM

# 12.11.2022
[https://leetcode.com/problems/find-median-from-data-stream/](https://leetcode.com/problems/find-median-from-data-stream/) hard

To find the median we can maintain two heaps: smaller and larger. One decreasing and one increasing.
Peeking the top from those heaps will give us the median.

```

    //  [5 2 0] [6 7 10]
    //  dec     inc
    //   ^ peek  ^ peek

```

```kotlin

class MedianFinder() {
    val queDec = PriorityQueue<Int>(reverseOrder())
    val queInc = PriorityQueue<Int>()
    fun addNum(num: Int) {
        if (queDec.size == queInc.size) {
            queInc.add(num)
            queDec.add(queInc.poll())
        } else {
            queDec.add(num)
            queInc.add(queDec.poll())
        }
    }

    fun findMedian(): Double = if (queInc.size == queDec.size)
            (queInc.peek() + queDec.peek()) / 2.0
        else 
            queDec.peek().toDouble()
}

```

Complexity: O(NlogN)
Memory: O(N)
# 11.11.2022
[https://leetcode.com/problems/remove-duplicates-from-sorted-array/](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) easy

Just do what is asked. Keep track of the pointer to the end of the "good" part.

```

    fun removeDuplicates(nums: IntArray): Int {
        var k = 0
        for (i in 1..nums.lastIndex) {
            if (nums[k] != nums[i]) nums[++k] = nums[i]
        }
        
        return k + 1
    }

```

Complexity: O(N)
Memory: O(1)

# 10.11.2022
[https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/) easy

Solution:

```

    fun removeDuplicates(s: String): String {
        val stack = Stack<Char>()
        s.forEach { c ->
            if (stack.isNotEmpty() && stack.peek() == c) {
                stack.pop()
            } else {
                stack.push(c)
            }
        }
        return stack.joinToString("")
    }

```

Explanation: Just scan symbols one by one and remove duplicates from the end.
Complexity: O(N)
Memory: O(N)

# 9.11.2022
[https://leetcode.com/problems/online-stock-span/](https://leetcode.com/problems/online-stock-span/) medium

So, we need to keep increasing sequence of numbers, increasing/decreasing stack will help.
Consider example, this is how decreasing stack will work
```bash
        // 100   [100-1]                            1
        // 80    [100-1, 80-1]                      1
        // 60    [100-1, 80-1, 60-1]                1
        // 70    [100-1, 80-1, 70-2] + 60           2
        // 60    [100-1, 80-1, 70-2, 60-1]          1
        // 75    [100-1, 80-1, 75-4] + 70-2+60-1    4
        // 85    [100-1, 85-6] 80-1+75-4            6

```

Solution:

```kotlin

class StockSpanner() {
    val stack = Stack<Pair<Int,Int>>()

    fun next(price: Int): Int {
        // 100   [100-1]                            1
        // 80    [100-1, 80-1]                      1
        // 60    [100-1, 80-1, 60-1]                1
        // 70    [100-1, 80-1, 70-2] + 60           2
        // 60    [100-1, 80-1, 70-2, 60-1]          1
        // 75    [100-1, 80-1, 75-4] + 70-2+60-1    4
        // 85    [100-1, 85-6] 80-1+75-4            6
       var span = 1
       while(stack.isNotEmpty() && stack.peek().first <= price) {
          span += stack.pop().second 
       } 
       stack.push(price to span)
       return span
    }

}

```

Complexity: O(N)
Memory: O(N)

# 8.11.2022
[https://leetcode.com/problems/make-the-string-great/](https://leetcode.com/problems/make-the-string-great/) easy

```kotlin

    fun makeGood(s: String): String {
        var ss = s.toCharArray()
        var finished = false
        while(!finished) {
            finished = true
            for (i in 0 until s.lastIndex) {
                if (ss[i] == '.') continue
                var j = i+1
                while(j <= s.lastIndex && ss[j] == '.') {
                    j++
                    continue
                }
                if (j == s.length) break
                
                var a = ss[i]
                var b = ss[j]
                if (a != b && Character.toLowerCase(a) == 
                        Character.toLowerCase(b)) {
                    ss[i] = '.'
                    ss[j] = '.'
                    finished = false
                }
            }
        }
        return ss.filter { it != '.' }.joinToString("")
    }

```

Explanation:
The simplest solution is just to simulate all the process, as input string is just 100 symbols.

Speed: O(n^2)
Memory: O(n)

# 7.11.2022
[https://leetcode.com/problems/maximum-69-number/](https://leetcode.com/problems/maximum-69-number/) easy

```kotlin

    fun maximum69Number (num: Int): Int {
        var n = num
        if (6666 <= n && n <= 6999) return num + 3000
        if (n > 9000) n -= 9000
        if (666 <= n && n <= 699) return num + 300
        if (n > 900) n -= 900
        if (66 <= n && n <= 69) return num + 30
        if (n > 90) n -= 90
        if (6 == n) return num + 3
        return num
    }

```

Explanation:
The simplest implementations would be converting to array of digits, replacing the first and converting back. 
However we can observe that numbers are in range 6-9999, so we can hardcode some logic.

Speed: O(1), Memory: O(1)

# 6.11.2022
[https://leetcode.com/problems/orderly-queue/](https://leetcode.com/problems/orderly-queue/) hard

```kotlin

    fun orderlyQueue(s: String, k: Int): String {
        val chrs = s.toCharArray()
        if (k == 1) {
            var smallest = s
            for (i in 0..s.lastIndex) {
                val prefix = s.substring(0, i)
                val suffix = s.substring(i)
                val ss = suffix + prefix
                if (ss.compareTo(smallest) < 0) smallest = ss
            }
            return smallest
        } else {
            chrs.sort()
            return String(chrs)
        }
    }

O(n^2)

```

Explanation:
One idea that come to my mind is: if k >= 2 then you basically can swap any adjacent elements. That means you can actually sort all the characters.

Speed: O(n^2), Memory: O(n)
    
# 6.11.2022
[https://leetcode.com/problems/word-search-ii/](https://leetcode.com/problems/word-search-ii/) hard

Solution [kotlin]

```kotlin

    class Node {
        val next = Array<Node?>(26) { null }
        var word: String?  = null
        operator fun invoke(c: Char): Node {
            val ind = c.toInt() - 'a'.toInt()
            if (next[ind] == null) next[ind] = Node()
            return next[ind]!!
        } 
        operator fun get(c: Char) = next[c.toInt() - 'a'.toInt()]
    }
    fun findWords(board: Array<CharArray>, words: Array<String>): List<String> {
        val trie = Node()
        
        words.forEach { w ->
            var t = trie
            w.forEach { t = t(it) }
            t.word = w
        }
        
        val result = mutableSetOf<String>()
        fun dfs(y: Int, x: Int, t: Node?, visited: MutableSet<Int>) {
           if (t == null || y < 0 || x < 0 
               || y >= board.size || x >= board[0].size 
               || !visited.add(100 * y + x)) return
           t[board[y][x]]?.let {
               it.word?.let {  result.add(it)  }
                dfs(y-1, x, it, visited)
                dfs(y+1, x, it, visited)
                dfs(y, x-1, it, visited)
                dfs(y, x+1, it, visited)
           }
           visited.remove(100 * y + x)
        }
        board.forEachIndexed { y, row ->
            row.forEachIndexed { x, c ->
                dfs(y, x, trie, mutableSetOf<Int>())
            }
        }
        return result.toList()
    }

```

Explanation:
Use trie + dfs
1. Collect all the words into the Trie
2. Search deeply starting from all the cells and advancing trie nodes
3. Collect if node is the word
4. Use set to avoid duplicates

Speed: O(wN + M), w=10, N=10^4, M=12^2 , Memory O(26w + N) 

# 4.11.2022
[https://leetcode.com/problems/reverse-vowels-of-a-string/](https://leetcode.com/problems/reverse-vowels-of-a-string/) easy

Solution [kotlin]

```kotlin

    fun reverseVowels(s: String): String {
        val vowels = setOf('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        var chrs = s.toCharArray()
        var l = 0
        var r = chrs.lastIndex
        while(l < r) {
            while(l<r && chrs[l] !in vowels) l++
            while(l<r && chrs[r] !in vowels) r--
            if (l < r) chrs[l] = chrs[r].also { chrs[r] = chrs[l] }
            r--
            l++
        }
        return String(chrs)
    }

```

Explanation:
Straightforward solution : use two pointers method and scan from the both sides.

Speed: O(N), Memory O(N)

# 3.11.2022
[https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/](https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/) medium

Solution [kotlin]

```kotlin

fun longestPalindrome(words: Array<String>): Int {
        var singles = 0
        var mirrored = 0
        var uneven = 0
        var unevenSum = 0
        val visited = mutableMapOf<String, Int>()
        words.forEach { w ->  visited[w] = 1 + visited.getOrDefault(w, 0) }
        visited.forEach { w, wCount ->
            if (w[0] == w[1]) {
                if (wCount %2 == 0) {
                    singles += wCount*2
                } else {
                    // a b -> a
                    // a b a -> aba 2a + 1b = 2 + 1
                    // a b a b -> abba 2a + 2b = 2+2
                    // a b a b a -> baaab 3a + 2b = 3+2
                    // a b a b a b -> baaab 3a + 3b = 3+2 (-1)
                    // a b a b a b a -> aabbbaa 4a+3b=4+3
                    // a b a b a b a b -> aabbbbaa 4a+4b=4+4
                    // 5a+4b = 2+5+2
                    // 5a+5b = 2+5+2 (-1)
                    // 1c + 2b + 2a = b a c a b
                    // 1c + 3b + 2a =
                    // 1c + 3b + 4a = 2a + 3b + 2a
                    // 5d + 3a + 3b + 3c = a b c 5d c b a = 11 
                    uneven++
                    unevenSum += wCount
                }
            } else {
                val matchingCount = visited[w.reversed()] ?:0
                mirrored += minOf(wCount, matchingCount)*2
            }
        }
        val unevenCount = if (uneven == 0) 0 else 2*(unevenSum - uneven + 1)
        return singles + mirrored + unevenCount
    }

```

Explanation:
This is a counting task, can be solved linearly.
There are 3 cases: 
1. First count mirrored elements, "ab" <-> "ba", they all can be included to the result
2. Second count doubled letters "aa", "bb". Notice, that if count is even, they also can be splitted by half and all included. 
3. The only edge case is uneven part. The law can be derived by looking at the examples

Speed: O(N), Memory O(N)

# 2.11.2022
[https://leetcode.com/problems/minimum-genetic-mutation/](https://leetcode.com/problems/minimum-genetic-mutation/) medium

Solution [kotlin]

```kotlin

    fun minMutation(start: String, end: String, bank: Array<String>): Int {
        val wToW = mutableMapOf<Int, MutableList<Int>>()
        fun searchInBank(i1: Int, w1: String) {
            bank.forEachIndexed { i2, w2 ->
                if (w1 != w2) {
                    var diffCount = 0
                    for (i in 0..7) {
                        if (w1[i] != w2[i]) diffCount++
                    }
                    if (diffCount == 1) {
                       wToW.getOrPut(i1, { mutableListOf() }).add(i2)
                       wToW.getOrPut(i2, { mutableListOf() }).add(i1)
                    }
                }
            }
        }
        bank.forEachIndexed { i1, w1 -> searchInBank(i1, w1) }
        searchInBank(-1, start)
        val queue = ArrayDeque<Int>()
        queue.add(-1)
        var steps = 0
        while(queue.isNotEmpty()) {
            repeat(queue.size) {
                val ind = queue.poll()
                val word = if (ind == -1) start else bank[ind]
                if (word == end) return steps
                wToW[ind]?.let { siblings ->
                    siblings.forEach { queue.add(it) }
                }
            }
            steps++
            if (steps > bank.size + 1) return -1
        }
        return -1
    }

```

Explanation:
1. make graph
2. BFS in it
3. stop search if count > bank, or we can use visited map

Speed: O(wN^2), Memory O(N)

# 1.11.2022
[https://leetcode.com/problems/where-will-the-ball-fall/](https://leetcode.com/problems/where-will-the-ball-fall/) medium

Solution [kotlin]

```kotlin

    fun findBall(grid: Array<IntArray>): IntArray {
        var indToBall = IntArray(grid[0].size) { it }
        var ballToInd = IntArray(grid[0].size) { it }
        grid.forEach { row ->
            var nextIndToBall = IntArray(grid[0].size) { -1 }
            var nextBallToInd = IntArray(grid[0].size) { -1 }
            for (i in 0..row.lastIndex) {
                val currBall = indToBall[i]
                if (currBall != -1) {
                    val isCorner = row[i] == 1 
                    &&  i<row.lastIndex 
                    && row[i+1] == -1
                    || row[i] == -1
                    && i > 0
                    && row[i-1] == 1
                    
                    val newInd = i + row[i]
                    if (!isCorner && newInd >= 0 && newInd <= row.lastIndex) {
                        nextIndToBall[newInd] = currBall
                        nextBallToInd[currBall] = newInd
                    } 
                }
            }
            indToBall = nextIndToBall
            ballToInd = nextBallToInd
        }
        return ballToInd
    }

```

Explanation:
This is a geometry problem, but seeing the pattern might help. We can spot that each row is an action sequence: -1 -1 -1 shifts balls left, and 1 1 1 shifts balls to the right. Corners can be formed only with -1 1 sequence.  

# 31.10.2022
[https://leetcode.com/problems/toeplitz-matrix/](https://leetcode.com/problems/toeplitz-matrix/) easy

Solution [kotlin]

```kotlin

    fun isToeplitzMatrix(matrix: Array<IntArray>): Boolean =
        matrix
        .asSequence()
        .windowed(2)
        .all { (prev, curr) -> prev.dropLast(1) == curr.drop(1) }

```

Explanation:
just compare adjacent rows, they must have an equal elements except first and last
