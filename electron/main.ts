import { stopCompletion } from "./completion"
import { registerTraining, selectTrainingEnvironment, stopTraining } from "./training"
import { app, BrowserWindow, dialog, ipcMain, Menu, nativeImage, shell, Tray } from 'electron'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const appIconPath = join(__dirname, process.env.VITE_DEV_SERVER_URL ? '../public/greatnn-cat.png' : '../dist/greatnn-cat.png')

let mainWindow: BrowserWindow | null = null
let tray: Tray | null = null
let isQuitting = false
let interfaceLocale: 'zh' | 'en' = 'en'

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'GreatNN',
    icon: appIconPath,
    webPreferences: {
      preload: join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https:') || url.startsWith('http:')) {
      shell.openExternal(url)
    }
    return { action: 'deny' }
  })

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL)
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(join(__dirname, '../dist/index.html'))
  }

  // Hide window instead of closing — keeps app running in background
  mainWindow.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault()
      mainWindow?.hide()
    }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

function updateDesktopMenus() {
  const environmentMenu: Electron.MenuItemConstructorOptions = {
    label: interfaceLocale === 'zh' ? '导入本地 Python 环境…' : 'Select Local Python Environment…',
    click: () => {
      if (!mainWindow) return
      mainWindow.show()
      void selectTrainingEnvironment(mainWindow, interfaceLocale).catch((error: unknown) => {
        const { message } = error instanceof Error ? error : { message: String(error) }
        if (mainWindow) {
          void dialog.showMessageBox(mainWindow, {
            type: "error",
            title: "Python",
            message: interfaceLocale === "zh" ? "无法加载 Python 环境" : "Could not load the Python environment",
            detail: message,
          })
        }
      })
    },
  }
  if (interfaceLocale === 'zh') {
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      { label: '文件', submenu: [environmentMenu, { type: 'separator' }, { role: 'quit', label: '退出' }] },
      { label: '编辑', submenu: [
        { role: 'undo', label: '撤销' },
        { role: 'redo', label: '重做' },
        { type: 'separator' },
        { role: 'cut', label: '剪切' },
        { role: 'copy', label: '复制' },
        { role: 'paste', label: '粘贴' },
        { role: 'delete', label: '删除' },
        { role: 'selectAll', label: '全选' },
      ] },
      { label: '视图', submenu: [
        { role: 'reload', label: '重新加载' },
        { role: 'forceReload', label: '强制重新加载' },
        { role: 'toggleDevTools', label: '开发者工具' },
        { type: 'separator' },
        { role: 'resetZoom', label: '重置缩放' },
        { role: 'zoomIn', label: '放大' },
        { role: 'zoomOut', label: '缩小' },
        { role: 'togglefullscreen', label: '切换全屏' },
      ] },
      { label: '窗口', submenu: [
        { role: 'minimize', label: '最小化' },
        { role: 'close', label: '关闭窗口' },
      ] },
    ]))
  } else {
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      { label: 'File', submenu: [environmentMenu, { type: 'separator' }, { role: 'quit' }] }, { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
    ]))
  }

  const contextMenu = Menu.buildFromTemplate([
    {
      label: interfaceLocale === 'zh' ? '显示 GreatNN' : 'Show GreatNN',
      click: () => {
        if (mainWindow) {
          mainWindow.show()
          mainWindow.focus()
        } else {
          createWindow()
        }
      },
    },
    { type: 'separator' },
    {
      label: interfaceLocale === 'zh' ? '退出' : 'Quit',
      click: () => {
        isQuitting = true
        app.quit()
      },
    },
  ])

  tray?.setContextMenu(contextMenu)
}

function createTray() {
  tray = new Tray(nativeImage.createFromPath(appIconPath).resize({ width: 20, height: 20 }))
  tray.setToolTip('GreatNN')

  // Click tray icon to toggle window
  tray.on('click', () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide()
      } else {
        mainWindow.show()
        mainWindow.focus()
      }
    } else {
      createWindow()
    }
  })
}

app.on('activate', () => {
  if (mainWindow) {
    mainWindow.show()
  } else {
    createWindow()
  }
})

app.on('before-quit', () => {
  stopCompletion()
  void stopTraining()
  isQuitting = true
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

ipcMain.on('nnmind:set-locale', (event, locale: unknown) => {
  if (event.sender !== mainWindow?.webContents || (locale !== 'zh' && locale !== 'en')) return
  if (locale === interfaceLocale) return
  interfaceLocale = locale
  updateDesktopMenus()
})

registerTraining(() => mainWindow, () => interfaceLocale)

app.whenReady().then(() => {
  if (process.platform === 'win32') app.setAppUserModelId('org.greatnn.desktop')
  interfaceLocale = app.getLocale().toLowerCase().startsWith('zh') ? 'zh' : 'en'
  createTray()
  updateDesktopMenus()
  createWindow()
})
